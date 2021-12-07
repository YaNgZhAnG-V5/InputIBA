dataset_type = 'CXRDataset'
data_root = '/content'

data = dict(
    data_loader=dict(batch_size=1, shuffle=True, num_workers=0),
    estimation=dict(
        type=dataset_type,
        path_to_images=data_root + '/NIH small', #BrixIAsmall
        label_path='/content/drive/MyDrive/Prak_MLMI/model/labels',
        fold='BBox',
        finding='Cardiomegaly',
    )
)