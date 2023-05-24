import json 

def read_config(hyper_params_config_file):
    f = open(hyper_params_config_file,"r")
    hyper_params = json.load(f)
    return hyper_params

def save_report(report, classification_report_file):
    report_object = json.dumps(report)
    freport = open(classification_report_file, "w")
    freport.write(report_object)