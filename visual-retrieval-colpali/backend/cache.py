from collections import OrderedDict


# Initialize LRU Cache
class LRUCache:
    def __init__(self, max_size=20):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]
