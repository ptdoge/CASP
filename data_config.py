
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = ""
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR-CD+':
            self.label_transform = "norm"
            self.root_dir = '/data/ptdoge/LEVIR-CD+'
        elif data_name == 'WHU-CD':
            self.label_transform = "norm"
            self.root_dir = '/data/ptdoge/WHU-CD+' # 
        elif data_name == 'GZ_CD':
            self.label_transform = "norm"
            self.root_dir = '/data/ptdoge/CD_Data_GZ'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

