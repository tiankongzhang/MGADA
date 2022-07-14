class Container(dict):

    def _call(self, name, *args, **kwargs):
        keys = list(self.keys())
        for key in keys:
            value = self[key]
            if hasattr(value, name):
                self[key] = getattr(value, name)(*args, **kwargs)
        return self

    def to(self, *args, **kwargs):
        return self._call('to', *args, **kwargs)

    def numpy(self):
        return self._call('numpy')
