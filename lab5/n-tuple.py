def find_isomorphic_pattern(pattern):
    a = board(list(range(16)))

    isomorphic_pattern = []
    for i in range(8):
        if (i >= 4):
            b = board( a.mirror().tile )
        else:
            b = board( a.tile )
        for _ in range(i%4):
            b = b.rotate()
        isomorphic_pattern.append(np.array(b.tile)[pattern])
        
    return isomorphic_pattern

class TuplesNet():
    def __init__(self, pattern, maxnum):
        self.V = np.zeros(([maxnum]*len(pattern)))
        self.pattern = pattern
        self.isomorphic_pattern = find_isomorphic_pattern(self.pattern)
        
    def getState(self, tile):
        return [tuple(np.array(tile)[p]) for p in self.isomorphic_pattern]
    
    def getValue(self, tile):
        S = self.getState(tile)
        V = [self.V[s] for s in S]
        V = sum(V)

        return V
    
    def setValue(self, tile, v, reset=False):
        S = self.getState(tile)
        v /= len(self.isomorphic_pattern)
        V = 0.0
        for s in S:
            self.V[s] += v
            V += self.V[s]
        return V