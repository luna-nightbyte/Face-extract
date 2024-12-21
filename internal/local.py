class Internal:
    def __init__(self,version,model_sel,input_dir,output_dir, csv_path):
        import internal.detection.face_detector as faceDet
        from internal import csv as i_csv
        self.face_detection = faceDet.Core(version, model_sel)
        self.csv = i_csv.CSV_class(csv_path)
        self.input = input_dir
        self.output = output_dir
        