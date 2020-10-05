__all__ = [
    'HNXCount'
]


class HNXCount():
    def __init__(self, init=0):
        self.init = init
        self.value = init

    def __call__(self):
        temp = self.value
        self.value += 1
        return temp
