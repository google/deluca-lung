EESchema Schematic File Version 4
EELAYER 30 0
EELAYER END
$Descr A4 11693 8268
encoding utf-8
Sheet 1 1
Title "Venti Zero"
Date "2021-05-11"
Rev "2.0"
Comp ""
Comment1 ""
Comment2 "Copyright 2021 Google LLC"
Comment3 "License: Apache 2.0"
Comment4 "Author: Daniel Suo"
$EndDescr
$Comp
L Connector_Generic:Conn_02x20_Odd_Even J1
U 1 1 5C77771F
P 3550 2100
F 0 "J1" H 3600 3217 50  0000 C CNN
F 1 "GPIO_CONNECTOR" H 3600 3126 50  0000 C CNN
F 2 "lib:PinSocket_2x20_P2.54mm_Vertical_Centered_Anchor" H 3550 2100 50  0001 C CNN
F 3 "~" H 3550 2100 50  0001 C CNN
	1    3550 2100
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR0101
U 1 1 5C777805
P 3150 3250
F 0 "#PWR0101" H 3150 3000 50  0001 C CNN
F 1 "GND" H 3155 3077 50  0001 C CNN
F 2 "" H 3150 3250 50  0001 C CNN
F 3 "" H 3150 3250 50  0001 C CNN
	1    3150 3250
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR0102
U 1 1 5C777838
P 4050 3250
F 0 "#PWR0102" H 4050 3000 50  0001 C CNN
F 1 "GND" H 4055 3077 50  0001 C CNN
F 2 "" H 4050 3250 50  0001 C CNN
F 3 "" H 4050 3250 50  0001 C CNN
	1    4050 3250
	1    0    0    -1  
$EndComp
Wire Wire Line
	3350 1600 3150 1600
Wire Wire Line
	3150 1600 3150 2400
Wire Wire Line
	3350 2400 3150 2400
Connection ~ 3150 2400
Wire Wire Line
	3150 2400 3150 3100
Wire Wire Line
	3350 3100 3150 3100
Connection ~ 3150 3100
Wire Wire Line
	3150 3100 3150 3250
Wire Wire Line
	3850 1400 4050 1400
Wire Wire Line
	4050 1400 4050 1800
Wire Wire Line
	3850 1800 4050 1800
Connection ~ 4050 1800
Wire Wire Line
	4050 1800 4050 2100
Wire Wire Line
	3850 2100 4050 2100
Connection ~ 4050 2100
Wire Wire Line
	3850 2600 4050 2600
Wire Wire Line
	4050 2100 4050 2600
Connection ~ 4050 2600
Wire Wire Line
	4050 2600 4050 2800
Wire Wire Line
	3850 2800 4050 2800
Connection ~ 4050 2800
Wire Wire Line
	4050 2800 4050 3250
$Comp
L power:+5V #PWR0104
U 1 1 5C777E01
P 4150 1100
F 0 "#PWR0104" H 4150 950 50  0001 C CNN
F 1 "+5V" H 4165 1273 50  0000 C CNN
F 2 "" H 4150 1100 50  0001 C CNN
F 3 "" H 4150 1100 50  0001 C CNN
	1    4150 1100
	1    0    0    -1  
$EndComp
Wire Wire Line
	3850 1200 4150 1200
Wire Wire Line
	4150 1200 4150 1100
Wire Wire Line
	3850 1300 4150 1300
Wire Wire Line
	4150 1300 4150 1200
Connection ~ 4150 1200
$Comp
L power:PWR_FLAG #FLG0103
U 1 1 5C77CEFA
P 4500 1100
F 0 "#FLG0103" H 4500 1175 50  0001 C CNN
F 1 "PWR_FLAG" H 4500 1274 50  0000 C CNN
F 2 "" H 4500 1100 50  0001 C CNN
F 3 "~" H 4500 1100 50  0001 C CNN
	1    4500 1100
	1    0    0    -1  
$EndComp
Wire Wire Line
	4150 1200 4500 1200
Wire Wire Line
	4500 1100 4500 1200
Text Label 2400 1300 0    50   ~ 0
GPIO2_SDA1
Text Label 2400 1400 0    50   ~ 0
GPIO3_SCL1
Text Label 2400 1500 0    50   ~ 0
GPIO4_GPIO_GCLK
Text Label 2400 1700 0    50   ~ 0
GPIO17_GEN0
Text Label 2400 1800 0    50   ~ 0
GPIO27_GEN2
Text Label 2400 1900 0    50   ~ 0
GPIO22_GEN3
Text Label 2400 2100 0    50   ~ 0
GPIO10_SPI_MOSI
Wire Wire Line
	2300 2100 3350 2100
Wire Wire Line
	2300 2200 3350 2200
Wire Wire Line
	2300 2300 3350 2300
Wire Wire Line
	2300 2500 3350 2500
Wire Wire Line
	2300 2600 3350 2600
Wire Wire Line
	2300 2700 3350 2700
Wire Wire Line
	2300 2800 3350 2800
Wire Wire Line
	2300 2900 3350 2900
Wire Wire Line
	2300 3000 3350 3000
Wire Wire Line
	2300 1900 3350 1900
Wire Wire Line
	2300 1800 3350 1800
Wire Wire Line
	2300 1700 3350 1700
Wire Wire Line
	2300 1500 3350 1500
Wire Wire Line
	2300 1400 3350 1400
Wire Wire Line
	2300 1300 3350 1300
Text Label 2400 2200 0    50   ~ 0
GPIO9_SPI_MISO
Text Label 2400 2300 0    50   ~ 0
GPIO11_SPI_SCLK
Text Label 2400 2500 0    50   ~ 0
ID_SD
Text Label 2400 2600 0    50   ~ 0
GPIO5
Text Label 2400 2700 0    50   ~ 0
GPIO6
Text Label 2400 2800 0    50   ~ 0
GPIO13
Text Label 2400 2900 0    50   ~ 0
GPIO19
Text Label 2400 3000 0    50   ~ 0
GPIO26
NoConn ~ 2300 1500
NoConn ~ 2300 1800
NoConn ~ 2300 1900
NoConn ~ 2300 2100
NoConn ~ 2300 2200
NoConn ~ 2300 2300
NoConn ~ 2300 2500
NoConn ~ 2300 2700
NoConn ~ 2300 2800
NoConn ~ 2300 2900
NoConn ~ 2300 3000
Text Label 4200 1600 0    50   ~ 0
GPIO15_RXD0
Text Label 4200 1700 0    50   ~ 0
GPIO18_GEN1
Text Label 4200 1900 0    50   ~ 0
GPIO23_GEN4
Text Label 4200 2000 0    50   ~ 0
GPIO24_GEN5
Text Label 4200 2200 0    50   ~ 0
GPIO25_GEN6
Text Label 4200 2300 0    50   ~ 0
GPIO8_SPI_CE0_N
Text Label 4200 2400 0    50   ~ 0
GPIO7_SPI_CE1_N
Wire Wire Line
	3850 2300 4900 2300
Wire Wire Line
	3850 2400 4900 2400
Text Label 4200 2500 0    50   ~ 0
ID_SC
Text Label 4200 2700 0    50   ~ 0
GPIO12
Text Label 4200 2900 0    50   ~ 0
GPIO16
Text Label 4200 3000 0    50   ~ 0
GPIO20
Text Label 4200 3100 0    50   ~ 0
GPIO21
Wire Wire Line
	3850 1500 4900 1500
Wire Wire Line
	3850 1600 4900 1600
Wire Wire Line
	3850 1700 4900 1700
Wire Wire Line
	3850 1900 4900 1900
Wire Wire Line
	3850 2000 4900 2000
Wire Wire Line
	3850 2200 4900 2200
Wire Wire Line
	3850 2500 4900 2500
Wire Wire Line
	3850 2700 4900 2700
Wire Wire Line
	3850 2900 4900 2900
Wire Wire Line
	3850 3000 4900 3000
NoConn ~ 4900 1500
NoConn ~ 4900 1600
NoConn ~ 4900 1700
NoConn ~ 4900 1900
NoConn ~ 4900 2000
NoConn ~ 4900 2200
NoConn ~ 4900 2300
NoConn ~ 4900 2400
NoConn ~ 4900 2500
NoConn ~ 4900 2900
NoConn ~ 4900 3000
NoConn ~ 4900 3100
Wire Wire Line
	3850 3100 4900 3100
$Comp
L Mechanical:MountingHole H1
U 1 1 5C7C4C81
P 10450 750
F 0 "H1" H 10550 796 50  0000 L CNN
F 1 "MountingHole" H 10550 705 50  0000 L CNN
F 2 "lib:MountingHole_2.7mm_M2.5_uHAT_RPi" H 10450 750 50  0001 C CNN
F 3 "~" H 10450 750 50  0001 C CNN
	1    10450 750 
	1    0    0    -1  
$EndComp
$Comp
L Mechanical:MountingHole H2
U 1 1 5C7C7FBC
P 10450 950
F 0 "H2" H 10550 996 50  0000 L CNN
F 1 "MountingHole" H 10550 905 50  0000 L CNN
F 2 "lib:MountingHole_2.7mm_M2.5_uHAT_RPi" H 10450 950 50  0001 C CNN
F 3 "~" H 10450 950 50  0001 C CNN
	1    10450 950 
	1    0    0    -1  
$EndComp
$Comp
L Mechanical:MountingHole H3
U 1 1 5C7C8014
P 10450 1150
F 0 "H3" H 10550 1196 50  0000 L CNN
F 1 "MountingHole" H 10550 1105 50  0000 L CNN
F 2 "lib:MountingHole_2.7mm_M2.5_uHAT_RPi" H 10450 1150 50  0001 C CNN
F 3 "~" H 10450 1150 50  0001 C CNN
	1    10450 1150
	1    0    0    -1  
$EndComp
$Comp
L Mechanical:MountingHole H4
U 1 1 5C7C8030
P 10450 1350
F 0 "H4" H 10550 1396 50  0000 L CNN
F 1 "MountingHole" H 10550 1305 50  0000 L CNN
F 2 "lib:MountingHole_2.7mm_M2.5_uHAT_RPi" H 10450 1350 50  0001 C CNN
F 3 "~" H 10450 1350 50  0001 C CNN
	1    10450 1350
	1    0    0    -1  
$EndComp
$Comp
L venti-zero-rescue:venti-rescue_ULN2003ADR-venti-cache-venti-cache U4
U 1 1 60660A2B
P 7800 3950
F 0 "U4" H 7800 5238 60  0000 C CNN
F 1 "ULN2003ADR" H 7800 5132 60  0000 C CNN
F 2 "lib:ULN2003A" H 7800 3890 60  0001 C CNN
F 3 "" H 7800 3950 60  0000 C CNN
	1    7800 3950
	1    0    0    -1  
$EndComp
Text Label 4200 1500 0    50   ~ 0
GPIO14_TXD0
Text Label 2300 1300 2    50   ~ 0
SDA
Text Label 2300 1400 2    50   ~ 0
SCL
Text Label 2300 2600 2    50   ~ 0
J_ALRT
Text Label 2300 1700 2    50   ~ 0
GPIO17
Text Label 4900 2700 0    50   ~ 0
GPIO12
$Comp
L power:GND #PWR0106
U 1 1 6068CB8F
P 4450 5450
F 0 "#PWR0106" H 4450 5200 50  0001 C CNN
F 1 "GND" H 4455 5277 50  0000 C CNN
F 2 "" H 4450 5450 50  0001 C CNN
F 3 "" H 4450 5450 50  0001 C CNN
	1    4450 5450
	1    0    0    -1  
$EndComp
$Comp
L power:+5V #PWR0107
U 1 1 6068D2A3
P 2250 3800
F 0 "#PWR0107" H 2250 3650 50  0001 C CNN
F 1 "+5V" H 2265 3973 50  0000 C CNN
F 2 "" H 2250 3800 50  0001 C CNN
F 3 "" H 2250 3800 50  0001 C CNN
	1    2250 3800
	1    0    0    -1  
$EndComp
Wire Wire Line
	4550 4500 4650 4500
Text Label 2750 4100 2    50   ~ 0
SDA
Text Label 2750 4000 2    50   ~ 0
SCL
Wire Wire Line
	2750 4300 2650 4300
Wire Wire Line
	2650 4300 2650 4700
Wire Wire Line
	2650 4700 4450 4700
Wire Wire Line
	4650 4700 4650 4500
Text Label 2750 4500 2    50   ~ 0
J_ALRT
$Comp
L venti-zero-rescue:venti-rescue_sensor_mini-p-venti-cache-venti-cache U1
U 1 1 60661AEB
P 3200 5150
F 0 "U1" H 3050 5500 50  0000 R CNN
F 1 "sensor_mini-p" H 3500 5450 50  0000 R CNN
F 2 "lib:MINI-P-Sensor-Horizontal" H 3200 5450 50  0001 C CNN
F 3 "" H 3200 5450 50  0001 C CNN
	1    3200 5150
	1    0    0    -1  
$EndComp
Wire Wire Line
	3800 5550 3800 5650
Wire Wire Line
	3800 5650 4200 5650
Wire Wire Line
	4200 5650 4200 5250
Wire Wire Line
	4200 5250 4450 5250
Wire Wire Line
	3150 5550 3150 5750
Wire Wire Line
	3150 5750 4300 5750
Wire Wire Line
	4300 5750 4300 5350
Wire Wire Line
	4300 5350 4450 5350
Wire Wire Line
	4450 5450 4450 5350
Connection ~ 4450 4700
Wire Wire Line
	4450 4700 4650 4700
Wire Wire Line
	4450 4700 4450 5250
Connection ~ 4450 5350
Connection ~ 4450 5250
Wire Wire Line
	4450 5250 4450 5350
Wire Wire Line
	3900 5550 3900 5850
Wire Wire Line
	3900 5850 4750 5850
Wire Wire Line
	4750 5850 4750 3900
Wire Wire Line
	4750 3900 4550 3900
Wire Wire Line
	4850 5950 4850 3800
Wire Wire Line
	4850 3800 4550 3800
Text Notes 2950 5150 0    50   ~ 0
Airway\nPressure\nSensor
$Comp
L venti-zero-rescue:venti-rescue_sensor_mini-p-venti-cache-venti-cache U3
U 1 1 60662D68
P 3850 5150
F 0 "U3" H 3700 5500 50  0000 R CNN
F 1 "sensor_mini-p" H 4150 5450 50  0000 R CNN
F 2 "lib:MINI-P-Sensor-Horizontal" H 3850 5450 50  0001 C CNN
F 3 "" H 3850 5450 50  0001 C CNN
	1    3850 5150
	1    0    0    -1  
$EndComp
Text Notes 3600 5150 0    50   ~ 0
Differential\nPressure\nSensor
Wire Wire Line
	3700 5550 3700 5650
Wire Wire Line
	3700 5650 2550 5650
Wire Wire Line
	2550 5650 2550 3800
Connection ~ 2550 3800
Wire Wire Line
	2550 3800 2750 3800
Wire Wire Line
	3250 5550 3250 5950
Wire Wire Line
	3250 5950 4850 5950
Wire Wire Line
	3050 5550 3050 5750
Wire Wire Line
	3050 5750 2450 5750
Wire Wire Line
	2450 5750 2450 3800
Connection ~ 2450 3800
Wire Wire Line
	2450 3800 2550 3800
$Comp
L power:+24V #PWR0109
U 1 1 6072530D
P 7100 2500
F 0 "#PWR0109" H 7100 2350 50  0001 C CNN
F 1 "+24V" H 7115 2673 50  0000 C CNN
F 2 "" H 7100 2500 50  0001 C CNN
F 3 "" H 7100 2500 50  0001 C CNN
	1    7100 2500
	1    0    0    -1  
$EndComp
Wire Wire Line
	7000 3250 7000 3450
Wire Wire Line
	7000 3850 7100 3850
Wire Wire Line
	7100 3650 7000 3650
Connection ~ 7000 3650
Wire Wire Line
	7000 3650 7000 3850
Wire Wire Line
	7100 3450 7000 3450
Connection ~ 7000 3450
Wire Wire Line
	7000 3450 7000 3550
Wire Wire Line
	7100 4050 7000 4050
Wire Wire Line
	7000 4050 7000 4250
Wire Wire Line
	7000 4450 7100 4450
Wire Wire Line
	7100 4250 7000 4250
Connection ~ 7000 4250
Wire Wire Line
	7000 4250 7000 4450
Wire Wire Line
	8500 3250 8600 3250
Wire Wire Line
	8600 3250 8600 3450
Wire Wire Line
	8600 3850 8500 3850
Wire Wire Line
	8500 4050 8600 4050
Wire Wire Line
	8600 4450 8500 4450
Wire Wire Line
	8500 4250 8600 4250
Connection ~ 8600 4250
Wire Wire Line
	8600 4250 8600 4450
Wire Wire Line
	8500 3450 8600 3450
Connection ~ 8600 3450
Wire Wire Line
	8500 3650 8600 3650
Connection ~ 8600 3650
Wire Wire Line
	8600 3650 8600 3750
Wire Wire Line
	7000 3550 6900 3550
Connection ~ 7000 3550
Wire Wire Line
	7000 3550 7000 3650
Text Label 6900 3550 2    50   ~ 0
GPIO17
Wire Wire Line
	6900 4250 7000 4250
Text Label 6900 4250 2    50   ~ 0
GPIO12
Wire Wire Line
	8600 4050 8600 4250
Wire Wire Line
	7100 3250 7000 3250
NoConn ~ 3350 2000
NoConn ~ 3350 1200
$Comp
L eec:206832-0601 J2
U 1 1 609B4AE9
P 6950 2650
F 0 "J2" H 6708 3015 50  0000 C CNN
F 1 "206832-0601" H 6708 2924 50  0000 C CNN
F 2 "lib:Molex-206832-0601-MFG" H 6950 3150 50  0001 L CNN
F 3 "https://www.molex.com/pdm_docs/sd/2068320601_sd.pdf" H 6950 3250 50  0001 L CNN
F 4 "No" H 6950 3350 50  0001 L CNN "automotive"
F 5 "Conn" H 6950 3450 50  0001 L CNN "category"
F 6 "Tin,Nickel" H 6950 3550 50  0001 L CNN "contact material"
F 7 "12.5A" H 6950 3650 50  0001 L CNN "current rating"
F 8 "Connectors" H 6950 3750 50  0001 L CNN "device class L1"
F 9 "Headers and Wire Housings" H 6950 3850 50  0001 L CNN "device class L2"
F 10 "unset" H 6950 3950 50  0001 L CNN "device class L3"
F 11 "CONN HEADER VERT 6POS 3MM" H 6950 4050 50  0001 L CNN "digikey description"
F 12 "WM26556-ND" H 6950 4150 50  0001 L CNN "digikey part number"
F 13 "https://www.molex.com/pdm_docs/sd/2068320601_sd.pdf" H 6950 4250 50  0001 L CNN "footprint url"
F 14 "10.16mm" H 6950 4350 50  0001 L CNN "height"
F 15 "yes" H 6950 4450 50  0001 L CNN "is connector"
F 16 "yes" H 6950 4550 50  0001 L CNN "is male"
F 17 "Yes" H 6950 4650 50  0001 L CNN "lead free"
F 18 "cc75f55849923b35" H 6950 4750 50  0001 L CNN "library id"
F 19 "Molex" H 6950 4850 50  0001 L CNN "manufacturer"
F 20 "538-206832-0601" H 6950 4950 50  0001 L CNN "mouser part number"
F 21 "6" H 6950 5050 50  0001 L CNN "number of contacts"
F 22 "2" H 6950 5150 50  0001 L CNN "number of rows"
F 23 "HDR6" H 6950 5250 50  0001 L CNN "package"
F 24 "3mm" H 6950 5350 50  0001 L CNN "pitch"
F 25 "Yes" H 6950 5450 50  0001 L CNN "rohs"
F 26 "+105°C" H 6950 5550 50  0001 L CNN "temperature range high"
F 27 "-40°C" H 6950 5650 50  0001 L CNN "temperature range low"
F 28 "600V" H 6950 5750 50  0001 L CNN "voltage rating"
	1    6950 2650
	1    0    0    -1  
$EndComp
$Comp
L eec:206832-0601 J2
U 2 1 609B8043
P 8800 3950
F 0 "J2" H 8372 3704 50  0000 R CNN
F 1 "206832-0601" H 8372 3795 50  0000 R CNN
F 2 "lib:Molex-206832-0601-MFG" H 8800 4450 50  0001 L CNN
F 3 "https://www.molex.com/pdm_docs/sd/2068320601_sd.pdf" H 8800 4550 50  0001 L CNN
F 4 "No" H 8800 4650 50  0001 L CNN "automotive"
F 5 "Conn" H 8800 4750 50  0001 L CNN "category"
F 6 "Tin,Nickel" H 8800 4850 50  0001 L CNN "contact material"
F 7 "12.5A" H 8800 4950 50  0001 L CNN "current rating"
F 8 "Connectors" H 8800 5050 50  0001 L CNN "device class L1"
F 9 "Headers and Wire Housings" H 8800 5150 50  0001 L CNN "device class L2"
F 10 "unset" H 8800 5250 50  0001 L CNN "device class L3"
F 11 "CONN HEADER VERT 6POS 3MM" H 8800 5350 50  0001 L CNN "digikey description"
F 12 "WM26556-ND" H 8800 5450 50  0001 L CNN "digikey part number"
F 13 "https://www.molex.com/pdm_docs/sd/2068320601_sd.pdf" H 8800 5550 50  0001 L CNN "footprint url"
F 14 "10.16mm" H 8800 5650 50  0001 L CNN "height"
F 15 "yes" H 8800 5750 50  0001 L CNN "is connector"
F 16 "yes" H 8800 5850 50  0001 L CNN "is male"
F 17 "Yes" H 8800 5950 50  0001 L CNN "lead free"
F 18 "cc75f55849923b35" H 8800 6050 50  0001 L CNN "library id"
F 19 "Molex" H 8800 6150 50  0001 L CNN "manufacturer"
F 20 "538-206832-0601" H 8800 6250 50  0001 L CNN "mouser part number"
F 21 "6" H 8800 6350 50  0001 L CNN "number of contacts"
F 22 "2" H 8800 6450 50  0001 L CNN "number of rows"
F 23 "HDR6" H 8800 6550 50  0001 L CNN "package"
F 24 "3mm" H 8800 6650 50  0001 L CNN "pitch"
F 25 "Yes" H 8800 6750 50  0001 L CNN "rohs"
F 26 "+105°C" H 8800 6850 50  0001 L CNN "temperature range high"
F 27 "-40°C" H 8800 6950 50  0001 L CNN "temperature range low"
F 28 "600V" H 8800 7050 50  0001 L CNN "voltage rating"
	2    8800 3950
	-1   0    0    1   
$EndComp
$Comp
L power:GND1 #PWR0103
U 1 1 60B3477D
P 8750 4850
F 0 "#PWR0103" H 8750 4600 50  0001 C CNN
F 1 "GND1" H 8755 4677 50  0000 C CNN
F 2 "" H 8750 4850 50  0001 C CNN
F 3 "" H 8750 4850 50  0001 C CNN
	1    8750 4850
	1    0    0    -1  
$EndComp
Wire Wire Line
	8500 4650 8750 4650
Wire Wire Line
	8750 4650 8750 4850
Wire Wire Line
	8750 4650 8900 4650
Wire Wire Line
	8900 4650 8900 3950
Connection ~ 8750 4650
Wire Wire Line
	8600 4250 8750 4250
Wire Wire Line
	6850 3050 6850 2850
Connection ~ 6850 2850
Wire Wire Line
	6850 2850 6850 2650
Wire Wire Line
	7100 2500 7100 2850
Wire Wire Line
	6850 2850 7100 2850
Connection ~ 7100 2850
Wire Wire Line
	7100 2850 7100 3050
Wire Wire Line
	2250 3800 2450 3800
$Comp
L venti-zero-rescue:venti-rescue_ADS1115IDGSR-venti-cache-venti-cache U2
U 1 1 6065B8B4
P 3650 4100
F 0 "U2" H 3650 4788 60  0000 C CNN
F 1 "ADS1115IDGSR" H 3650 4682 60  0000 C CNN
F 2 "lib:ADS1115" H 3650 4040 60  0001 C CNN
F 3 "" H 3650 4100 60  0000 C CNN
	1    3650 4100
	1    0    0    -1  
$EndComp
NoConn ~ 4550 4100
NoConn ~ 4550 4200
Wire Wire Line
	8600 3450 8600 3550
Wire Wire Line
	8750 3550 8900 3550
Wire Wire Line
	8750 3550 8750 4250
Wire Wire Line
	8600 3550 8600 3650
Wire Wire Line
	8600 3750 8900 3750
Wire Wire Line
	8600 3650 8600 3750
Connection ~ 8600 3550
Wire Wire Line
	8600 3550 8600 3650
Connection ~ 8600 3750
Wire Wire Line
	8600 3750 8600 3850
$EndSCHEMATC
