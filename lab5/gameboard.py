class board:
    """simple implementation of 2048 puzzle"""
    
    def __init__(self, tile = None, max_number=15):
        self.tile = tile if tile is not None else [0] * 16
        self.max_num = max_number
    
    def __str__(self):
        state = '+' + '-' * 24 + '+\n'
        for row in [self.tile[r:r + 4] for r in range(0, 16, 4)]:
            state += ('|' + ''.join('{0:6d}'.format((1 << t) & -2) for t in row) + '|\n')
        state += '+' + '-' * 24 + '+'
        return state
    
    def mirror(self):
        return board([self.tile[r + i] for r in range(0, 16, 4) for i in reversed(range(4))])
    
    def transpose(self):
        return board([self.tile[r + i] for i in range(4) for r in range(0, 16, 4)])
    
    def rotate(self):
        return board([self.tile[4*(3-(i%4)) + (i//4)] for i in range(16)])
    
    def left(self):
        move, score = [], 0
        for row in [self.tile[r:r+4] for r in range(0, 16, 4)]:
            row, buf = [], [t for t in row if t]
            while buf:
                if len(buf) >= 2 and buf[0] is buf[1]:
                    buf = buf[1:]
                    buf[0] += 1
                    score += 1 << buf[0]
                row += [buf[0]]
                buf = buf[1:]
            move += row + [0] * (4 - len(row))
        return board(move), score if move != self.tile else -1
    
    def right(self):
        move, score = self.mirror().left()
        return move.mirror(), score
    
    def up(self):
        move, score = self.transpose().left()
        return move.transpose(), score
    
    def down(self):
        move, score = self.transpose().right()
        return move.transpose(), score
    
    def popup(self):
        tile = self.tile[:]
        empty = [i for i, t in enumerate(tile) if not t]
        tile[random.choice(empty)] = random.choice([1] * 9 + [2])
        return board(tile)
    
    def end(self):
        tile = self.tile[:]
        empty = [i for i, t in enumerate(tile) if not t]
        
        count_max_num = np.count_nonzero(self.max_num == np.array(tile))
        return len(empty) == 0 or count_max_num > 0