from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'RT': Dataset_Custom
}


def data_provider(args, flag):
    """
    数据加载器函数，根据不同任务类型和数据集选择合适的数据处理方式
    
    参数:
        args: 参数对象，包含数据加载和处理的各种配置
            - data: 数据集类型名称，如'ETTh1', 'custom', 'm4'等
            - embed: 时间特征编码方式
            - batch_size: 批处理大小
            - freq: 时间频率，如'h'(小时), 'd'(天)等
            - num_workers: 数据加载线程数
            - task_name: 任务类型
            - root_path: 数据根目录
            - seq_len: 输入序列长度
            - label_len: 标签长度
            - pred_len: 预测序列长度
            - features: 特征类型，如'M'(多变量预测多变量), 'S'(单变量预测单变量), 'MS'(多变量预测单变量)
            - target: 目标特征(在'S'或'MS'任务中)
            - seasonal_patterns: M4数据集的季节性模式
            - data_path: 数据文件路径
        flag: 数据集状态标志，如'train', 'test', 'val'
    
    返回:
        data_set: Dataset对象，包含处理后的数据集
        data_loader: DataLoader对象，用于批量加载数据
    """
    # 根据args.data参数选择对应的数据集类
    Data = data_dict[args.data]
    # 设置时间编码参数，如果args.embed为'timeF'则为1，否则为0
    timeenc = 0 if args.embed != 'timeF' else 1
    
    # 根据flag参数设置是否打乱数据
    # 测试阶段不打乱数据，训练和验证阶段打乱数据
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False  # 默认不丢弃最后一个不完整的批次
    batch_size = args.batch_size  # 批处理大小
    freq = args.freq  # 时间频率

    # 异常检测任务的特殊处理
    if args.task_name == 'anomaly_detection':
        drop_last = False
        # 创建异常检测用的数据集对象
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,  # 使用seq_len作为窗口大小
            flag=flag,
        )
        print(flag, len(data_set))
         # 创建数据加载器
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    # 分类任务的特殊处理
    elif args.task_name == 'classification':
        drop_last = False
        # 创建分类用的数据集对象
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )
        # 创建数据加载器，使用collate_fn处理不等长序列
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    # 预测任务(默认)的处理
    else:
        # M4数据集的特殊处理
        if args.data == 'm4':
            drop_last = False
        # 创建预测任务的数据集对象
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],  # 设置序列长度相关参数
            features=args.features,  # 特征类型(M/S/MS)
            target=args.target,  # 目标特征
            timeenc=timeenc,  # 时间编码方式
            freq=freq, # 时间频率
            seasonal_patterns=args.seasonal_patterns  # 季节性模式(用于M4数据集)
        )
        print(flag, len(data_set))

        # 创建数据加载器
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
