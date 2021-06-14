# zymomonas_modeling
___substrate_uptake_kinetics.py___ estimates the kinetic parameters of glucose facilitator protein (glf) which transports both glucose and xylose with competitive inhibition of each other.   
Required input file: measured_kinetics.xlsx   
   
___constrained_MDF_and_EPC.py___ performs thermodynamics analysis which maximize the minimal driving force and enzyme protein cost analysis which minimize the totol enzyme protein cost of a given pathway with additional metabolite concentration constraints. This is as upgraded version of the functions in [PathPaser](https://github.com/Chaowu88/PathParser).   
Required input file: glucose_utilization_pathway.tsv or xylose_utilization_pathway.tsv
