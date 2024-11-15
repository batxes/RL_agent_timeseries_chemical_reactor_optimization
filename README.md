A producer of chemicals. We have a timeseries for this year, resolution is 1h, the data is available for 6 reactors, so in total 37728 samples. Each sample has a number of values, e.g. for reactor 2:

2|CB 	2|Erdgas 	2|Konst.Stufe 	2|Perlwasser 	2|Regelstufe 	2|Sorte 	2|V-Luft 	2|VL Temp 	2|Fuelöl 	2|Makeöl 	2|Makeöl|Temperatur 	2|Makeöl|Ventil 	2|CCT 	2|CTD 	2|FCC 	2|SCT 	2|C 	2|H 	2|N 	2|O 	2|S
R2 CO2
R2 SO2
and there are some values that are common for all 6 reactors:
KD|Dampfmenge 	KD|Restgasmenge 	KD|NOx 	KD|Rauchgasmenge 	KD|SO2 	KE|Dampfmenge 	KE|Restgasmenge 	KE|NOx 	KE|Rauchgasmenge 	KE|SO2

most interesting would be to try to predict output values from the input values. Important outputs are:
- CB (the product)
- SO2
- CO2
- Dampfmenge (steam)
- Rauchgasmenge (tail gas, all reactors combined)

input is basically everything else except the values for all reactors:
KD|Dampfmenge 	KD|Restgasmenge 	KD|NOx 	KD|Rauchgasmenge 	KD|SO2 	KE|Dampfmenge 	KE|Restgasmenge 	KE|NOx 	KE|Rauchgasmenge 	KE|SO2