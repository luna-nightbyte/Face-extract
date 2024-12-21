import csv
class CSV_class:
    def __init__(self,path):
        self.path = path
        self.file = None
        
    def open(self): 
        if self.is_open():
            return f"{self.path} already open" 
        try:
            self.file = open(self.path, mode="a", newline="")
            
        except Exception:
            self.path = None
            pass
        return None
    def close(self):  
        if self.file is not None:
            self.file.close()
            self.file = None  
        else:
            return f"{self.path} not open"
        return None
            
    def is_open(self):  
        return self.file is not None
            
    def write(self,data):  
        if self.path is None:
            return f"writer not properly initialyzed!"
        if self.is_open():
            writer = csv.writer(self.file)
            writer.writerow(data)
            return f"wrote to file"
        else:
            self.open()
            self.write(data=data)
            return None