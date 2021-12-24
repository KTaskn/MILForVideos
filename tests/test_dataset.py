from dataset import DataSet

def test_loader():
    images = list(range(100))
    ds = DataSet(images, F=5, func_extract=lambda x: x)
    assert ds.__getitem__(0) == [0, 1, 2, 3, 4]
    assert ds.__getitem__(1) == [5, 6, 7, 8, 9]
    assert ds.__getitem__(19) == [95, 96, 97, 98, 99]
    
    # 端数の場合
    images = list(range(104))
    ds = DataSet(images, F=5, func_extract=lambda x: x)
    assert ds.__getitem__(0) == [0, 1, 2, 3, 4]
    assert ds.__getitem__(1) == [5, 6, 7, 8, 9]
    assert ds.__getitem__(19) == [95, 96, 97, 98, 99]
    assert ds.__getitem__(20) == [99, 100, 101, 102, 103]