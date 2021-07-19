import deluca


def convert_sim(sim):
    sim.__class__ = deluca.lung.environments.StitchedSim
    sim.inspiratory_model.__class__ = deluca.lung.utils.sim.nn.InspiratoryModel
    sim.inspiratory_model.default_model.__class__ = deluca.lung.utils.sim.nn.SNN

    for key in sim.inspiratory_model.boundary_dict:
        if sim.inspiratory_model.boundary_dict[key].__class__.__name__ == "ConstantModel":
            sim.inspiratory_model.boundary_dict[
                key
            ].__class__ = deluca.lung.utils.sim.nn.ConstantModel
        elif (
            sim.inspiratory_model.boundary_dict[key].__class__.__name__ == "RegressionBoundaryModel"
        ):
            sim.inspiratory_model.boundary_dict[
                key
            ].__class__ = deluca.lung.utils.sim.nn.RegressionBoundaryModel
        elif sim.inspiratory_model.boundary_dict[key].__class__.__name__ == "ShallowBoundaryModel":
            sim.inspiratory_model.boundary_dict[
                key
            ].__class__ = deluca.lung.utils.sim.nn.ShallowBoundaryModel
        elif sim.inspiratory_model.boundary_dict[key].__class__.__name__ == "SNN":
            sim.inspiratory_model.boundary_dict[key].__class__ = deluca.lung.utils.sim.nn.SNN
        else:
            print("ERROR!!!")

    u_mean, u_std = sim.u_scaler.mean, sim.u_scaler.std
    sim.u_scaler = deluca.lung.utils.core.TorchStandardScaler()
    sim.u_scaler.mean, sim.u_scaler.std = u_mean, u_std

    p_mean, p_std = sim.p_scaler.mean, sim.p_scaler.std
    sim.p_scaler = deluca.lung.utils.core.TorchStandardScaler()
    sim.p_scaler.mean, sim.p_scaler.std = p_mean, p_std

    return sim
