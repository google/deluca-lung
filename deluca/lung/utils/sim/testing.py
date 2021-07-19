from deluca.lung.utils.data.analyzer import Analyzer
from deluca.lung.utils.data.munger import Munger
from deluca.lung.controllers._predestined import Predestined
from deluca.lung.controllers._impulse import Impulse
from deluca.lung.utils.scripts.run_controller import run_controller, run_controller_scan

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
import pickle
import tqdm


def open_loop_test(sim, munger, key="test", abort=70, use_tqdm=False, scan=False, **kwargs):
    # run u_ins, get pressures, check with true trajectory

    if isinstance(munger, str):
        munger = pickle.load(open(munger, "rb"))

    test_summary = {}

    all_runs = []

    for test_trajectory in tqdm.tqdm(munger.splits[key]):
        test_u_ins, test_pressures = test_trajectory
        T = len(test_u_ins)
        controller = Predestined.create(u_ins=test_u_ins)
        if not scan:
            run_data = run_controller(
                controller, R=None, C=None, T=T, abort=abort, use_tqdm=use_tqdm, env=sim, **kwargs
            )
        else:
            run_data = run_controller_scan(
                controller, R=None, C=None, T=T, abort=abort, use_tqdm=use_tqdm, env=sim, **kwargs
            )

        analyzer = Analyzer(run_data)

        # pedantic copying + self-documenting names...
        preds = analyzer.pressure.copy()
        truth = test_pressures.copy()

        all_runs.append((preds, truth))

    test_summary["all_runs"] = all_runs
    populate_resids(test_summary)

    return test_summary

def forecasting_test(sim, munger_path, key="test", abort=70, use_tqdm=False, **kwargs):
    munger = Munger.load(munger_path)

    test_summary = {}
    all_runs = []

    with torch.no_grad():
        for test_trajectory in munger.splits[key]:
            test_u_ins, test_pressures = test_trajectory
            test_u_ins = sim.tensor(test_u_ins)
            test_pressures_tensor = sim.tensor(test_pressures)

            T = len(test_u_ins)

            for k in range(T):  # feed k steps of pressure history
                sim.reset()
                sim.state["t_in"] = max(k - 1, 0)
                if k > 0:
                    sim.state["u_history"] = list(sim.u_scaler.transform(test_u_ins[: k - 1]))
                    sim.state["p_history"] = list(sim.p_scaler.transform(test_pressures_tensor[:k]))
                    sim.sync_unscaled_pressure()

                forecasts = []  # predicted pressure trajectory

                for t in range(k - 1, T):  # feed the rest from generation
                    if t == -1:  # with no pressure feedback, p[0] is a prediction
                        forecasts.append(sim.pressure.item())
                    else:  # predict next p
                        u_in = test_u_ins[t]
                        sim.step(u_in, 0)
                        if t < T - 1:  # last forecast is available but not benchmarkable
                            forecasts.append(sim.pressure.item())

                # model's k-step forecast residuals
                preds = np.array(forecasts)
                truth = test_pressures[k:].copy()
                all_runs.append((preds, truth))

    test_summary["all_runs"] = all_runs
    populate_resids(test_summary)

    return test_summary


def populate_resids(test_summary, key="all_runs", l1_out="mae", l2_out="rmse"):
    all_runs = test_summary[key]

    Tmax = max(map(lambda x: len(x[0]), all_runs))
    l1, l2, counts = np.zeros(Tmax), np.zeros(Tmax), np.zeros(Tmax)

    for preds, truth in all_runs:
        resids = truth - preds
        T = len(resids)
        l1[:T] += abs(resids)
        l2[:T] += resids ** 2
        counts[:T] += 1

    test_summary[l1_out] = l1 / counts
    test_summary[l2_out] = (l2 / counts) ** 0.5


def collate_boundary(all_runs, first_k=5):
    all_preds = []
    all_truth = []

    for preds, truth in all_runs:
        if len(preds) < first_k:
            continue
        all_preds.append(preds[:first_k])
        all_truth.append(truth[:first_k])

    return np.array(all_preds), np.array(all_truth)


def plot_spray(all_runs, **kwargs):
    if len(all_runs) < 1000:
        indices = np.arange(len(all_runs))
    else:  # subsample if there are too many runs
        indices = np.random.choice(len(all_runs), 1000)

    for i in indices:
        preds, truth = all_runs[i]
        plt.plot(truth - preds, "gray", alpha=0.1)

    plt.xlabel("# simulated pressures")
    plt.grid()


def plot_error_metric(errors, **kwargs):
    plt.plot(errors, "b", marker="o", markersize=4, lw=1, **kwargs)
    for i, err in enumerate(errors):
        if i % 2 == 0:
            plt.annotate(f"{err:.2f}", (i, err + 0.2), color="b", ha="center", va="bottom")
        else:
            plt.annotate(f"{err:.2f}", (i, err - 0.2), color="b", ha="center", va="top")


def plot_sim_diagnostics(open_loop_summary=None, forecast_summary=None, run=None):
    plt.rc("figure", figsize=(8, 8))

    plt.subplot(211)  ##### open-loop test
    for preds, truth in open_loop_summary["all_runs"]:
        plt.plot(truth - preds, "gray", alpha=0.1)
    plt.title("open-loop test")
    plt.ylabel("residual")
    plot_spray(open_loop_summary["all_runs"])
    plot_error_metric(open_loop_summary["mae"])

    #     plt.subplot(132)  ##### forecasting test
    #     plt.title('forecasting test')
    #     plot_spray(forecast_summary['all_runs'])
    #     plot_error_metric(forecast_summary['mae'])

    plt.subplot(212)
    plt.title("pressure trajectories")
    plot_run(run)

    #### boundary residual plots
    #     boundary_preds, boundary_truth = collate_boundary(open_loop_summary['all_runs'])
    #     for i in range(5):
    #         plt.subplot(2,5,6+i)
    #         if i == 0:
    #             plt.ylabel('predicted pressure')
    #         plt.xlabel(f'true pressure (t={i+1})')
    #         plt.scatter(boundary_truth[:,i], boundary_preds[:,i], s=2, color='b')
    #         plt.grid()

    #         x_min, x_max = plt.xlim()
    #         y_min, y_max = plt.ylim()
    #         z = [min(x_min,y_min), max(x_max,y_max)]
    #         plt.plot(z, z, 'gray')

    plt.tight_layout()


def default_sim_diagnostics(sim, munger_path, open_loop_summary=None, forecast_summary=None):
    # full eval protocol, for the exceptionally lazy
    open_loop_summary = open_loop_summary or open_loop_test(sim, munger_path)
    forecast_summary = forecast_summary or forecasting_test(sim, munger_path)

    plot_sim_diagnostics(open_loop_summary, forecast_summary)


def plot_intrinsic_uncertainty(
    munger_path,
    figsize=(4, 4),
    x1_idx=("u", 0),
    x2_idx=None,
    y_idx=("p", 1),
    key="test",
    s=5,
    alpha=0.85,
):
    # munger = Munger.load(munger_path)
    munger = pickle.load(open(munger_path, "rb"))
    plt.rc("figure", figsize=figsize)

    x1 = []
    x2 = []
    y = []

    for trajectory in munger.splits[key]:
        u, p = trajectory
        x1.append(locals()[x1_idx[0]][x1_idx[1]])
        if x2_idx is not None:
            x2.append(locals()[x2_idx[0]][x2_idx[1]])
        y.append(locals()[y_idx[0]][y_idx[1]])

    if x2_idx is None:
        plt.scatter(x1, y, s=s, alpha=alpha)

        plt.ylabel(f"{y_idx}")
    else:
        # ys = np.zeros((len(x1), len(x2)))
        # for i in x1:
        contours = plt.tricontour(x1, x2, y, s=s, alpha=alpha)
        plt.ylabel(f"{x2_idx}")
        colorbar = plt.colorbar()
        plt.clabel(contours, inline=True, fontsize=8)
        colorbar.set_label(f"{y_idx}", rotation=90)

    plt.title("Intrinsic uncertainty")
    plt.xlabel(f"{x1_idx}")


def get_run(sim, munger_path, key="test", abort=70, use_tqdm=False, **kwargs):
    # run u_ins, get pressures, check with true trajectory

    munger = Munger.load(munger_path)

    test_trajectory = munger.splits[key][0]

    test_u_ins, test_pressures = test_trajectory
    T = len(test_u_ins)

    controller = Predestined(test_u_ins, np.zeros_like(test_u_ins))

    run_data = run_controller(controller, T=T, abort=abort, use_tqdm=use_tqdm, env=sim, **kwargs)

    analyzer = Analyzer(run_data)

    # pedantic copying + self-documenting names...
    preds = analyzer.pressure.copy()
    truth = test_pressures.copy()

    return (preds, truth)


def plot_run(run, **kwargs):
    preds, truth = run
    plt.plot(truth, "blue", alpha=1.0, label="true trajectory")
    plt.plot(preds, "orange", alpha=1.0, label="simulated trajectory")
    plt.legend(loc="lower right")

    plt.xlabel("# time steps")
    plt.ylabel("pressure (cm H2O)")
    plt.grid()


def plot_impulse_responses(
    impulses=np.arange(0, 101, 10),
    zero=5,
    start=0.5,
    end=0.65,
    ylim=100,
    dt=0.03,
    T=100,
    use_tqdm=False,
    abort=70,
    **kwargs,
):
    analyzers = []

    for impulse in impulses:
        ir = Impulse(impulse, start, end)

        # TODO: UPDATE SELF.RUN
        run_data = self.run(ir, dt=dt, T=T, PEEP=zero, use_tqdm=use_tqdm, abort=abort, **kwargs)
        analyzer = Analyzer(run_data)
        analyzers.append(analyzer)

    loss = 0
    colors = plt.cm.winter(np.linspace(0, 1, len(analyzers)))
    for i, analyzer in enumerate(analyzers):
        plt.plot(analyzer.tt, analyzer.pressure, color=colors[-i - 1])
        if impulses[0] == 0 and i == 0:
            loss += np.abs(analyzer.pressure - zero).mean()

    print(f"MAE for zero response: {loss}")

    plt.ylim(0, ylim)
    plt.show()

    return analyzers


def plot_pids(self):
    plt.rc("figure", figsize=(8, 2))

    for coeff in np.linspace(0.0, 0.5, 20):
        pid = PID([coeff, 0.5 - coeff, 0], waveform=BreathWaveform((5, 35)))
        self.reset()

        # TODO: REMOVE SELF
        result = self.run(pid, T=333, abort=100)
        analyzer = Analyzer(result)

        cmap = matplotlib.cm.get_cmap("rainbow")
        plt.plot(analyzer.tt, analyzer.pressure, "b", c=cmap(2 * coeff))
