from random import sample

BOARD = frozenset({
    (0,0),(0,1),(0,2),
    (1,0),(1,1),(1,2),
    (2,0),(2,1),(2,2),
        })

WINNING = {
    frozenset({(0,0),(0,1),(0,2)}),
    frozenset({(1,0),(1,1),(1,2)}),
    frozenset({(2,0),(2,1),(2,2)}),
    frozenset({(0,0),(1,0),(2,0)}),
    frozenset({(0,1),(1,1),(2,1)}),
    frozenset({(0,2),(1,2),(2,2)}),
    frozenset({(0,0),(1,1),(2,2)}),
    frozenset({(0,2),(1,1),(2,0)}),
        }

class State:
    def __init__(self, x: frozenset, o: frozenset):
        self.x: frozenset = x
        self.o: frozenset = o
        self.free = BOARD - self.x - self.o

    def __str__(self):
        s = ""
        for i in range(3):
            for j in range(3):
                if (i,j) in self.x: s+='x'
                elif (i,j) in self.o: s+='o'
                else: s+='.'
            s+='\n'
        return s
    
    def __eq__(self, other):
        return (self.x, self.o, self.free) == (other.x, other.o, other.free)
    
    def __hash__(self):
        return hash((self.x, self.o, self.free))
    
    def swap(self):
        return State(self.o, self.x)

    
def valid(state: State):
    x_wins = False
    o_wins = False
    clash = False
    for w in WINNING:
        if w.issubset(state.x):
            x_wins = True
        if w.issubset(state.o):
            o_wins = True
    if x_wins and o_wins: clash = True
    return len(state.x.intersection(state.o)) == 0 and len(state.o) - len(state.x) in {0,1} and clash == False

def win(state: State):
    for w in WINNING:
        if w.issubset(state.x): return 1
        if w.issubset(state.o): return -1
    return 0

def next_cell(cell):
    if cell[1] < 2: return (cell[0], cell[1]+1)
    elif cell[0] < 2: return (cell[0]+1, 0)
    return None

state_values = dict()
def init_sv(state_values, state: State, cell):
    if cell == None:
        if valid(state): state_values[state] = win(state)
    else:
        init_sv(state_values, State(state.x.union({cell}), state.o), next_cell(cell))
        init_sv(state_values, State(state.x, state.o.union({cell})), next_cell(cell))
        init_sv(state_values, State(state.x, state.o), next_cell(cell))

init_sv(state_values, State(frozenset(), frozenset()), (0,0))

# value iteration...
delta = 1
epsilon = 0.001
while delta > epsilon:
    delta = 0
    for state, value in state_values.items():
        if win(state) != 0 or len(state.free) == 0: continue
        best_successor = min([State(state.x.union({cell}), state.o).swap() for cell in state.free], key=lambda s: state_values[s])
        if win(best_successor) == 0 and len(best_successor.free) != 0:
            next_value = min([state_values[State(best_successor.x.union({cell}), best_successor.o).swap()] for cell in best_successor.free])
        else: next_value = win(best_successor.swap())
        if abs(next_value - value) > delta: delta = abs(next_value - value)
        state_values[state] = next_value


# play a game...
def get_move(state):
    x = int(input("your move (x): "))
    y = int(input("your move (y): "))
    while (x,y) not in state.free:
        x = int(input("your move (x): "))
        y = int(input("your move (y): "))
    return (x,y)

start = 0
state = State(frozenset(), frozenset())
if start == 1:
    state = State(state.x, state.o.union({get_move(state)}))
while win(state) == 0:
    move = min(sample([move for move in state.free], k=len(state.free)), key=lambda m: state_values[State(state.x.union({m}), state.o).swap()])
    state = State(state.x.union({move}), state.o)
    if win(state) != 0: break
    print(state)
    state = State(state.x, state.o.union({get_move(state)}))