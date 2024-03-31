import time, math, numpy as np, random, itertools
from game_rules import constants as c
from game_rules import game_logic as game
from math import sqrt, log
from ai_algorithms import a_star as a


class Node:
    def __init__(self, board, last_player, parent=None) -> None:
        self.board = board
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0  
        self.current_player = 1 if last_player == 2 else 2

    def __str__(self) -> str:
        string = "Vitórias: " + str(self.wins) + '\n'
        string += "Total: " + str(self.visits) + '\n'
        string += "Pontuação: " + str(self.ucb()) + '\n'
        string += "Probabilidade de vitória: " + str(self.score()) + '\n'
        return string

    def add_children(self) -> None:
        """add each possible move to a list of possible children for the current node/state"""
        if game.available_moves(self.board) == -1: return   # se não existirem jogadas possíveis, não adiciona nada
        if len(self.children) != 0: return
        for col in game.available_moves(self.board):  # itera sobre todas as colunas possíveis a serem jogadas
            if self.current_player == c.HUMAN_PIECE:  copy_board = game.simulate_move(self.board, c.AI_PIECE, col)
            else: copy_board = game.simulate_move(self.board, c.HUMAN_PIECE, col)    # cria uma cópia do tabuleiro atual e adiciona a nova jogada a ele
            self.children.append((Node(board=copy_board, last_player=self.current_player, parent=self), col))  # adiciona cada tabuleiro gerado a uma lista, junto com a identificação da coluna que o gerou

    def select_children(self):
        if (len(self.children) > 4):
            return random.sample(self.children, 4)
        return self.children
    

    def ucb(self) -> float:
        """calculate the Upper Confidence Bound of the node"""
        if self.visits == 0: return float('inf')
        exploitation = self.wins / self.visits
        exploration = sqrt(2) * sqrt(2 * log(self.parent.visits / self.visits, math.e)) if self.parent else 0
        return exploitation + exploration
    
    def score(self) -> float:
        """calculate the score of the node"""
        if self.visits == 0: return 0
        return self.wins / self.visits




class MCTS:
    def __init__(self, root: Node) -> None:
        self.root = root
        self.best_node = root   # melhor nó para ser simulado = nó com maior ucb
        self.best_score = 0     # ucb do melhor nó a ser simulado

    def start(self, max_time: int):
        self.root.add_children()
        for child in self.root.children:
            for _ in range(6):
                result = self.rollout(child[0])
                self.back_propagation(child[0], result)
        return self.search(max_time)
            

    def update_best_node(self, cur_node: Node):
        ucb = cur_node.ucb()
        if ucb > self.best_score:
            self.best_node = cur_node
            self.best_score = ucb


    def search(self, max_time: int) -> int:
        """iterate through the tree of possible plays"""
        start_time = time.time()
        while time.time() - start_time < max_time:  # interrompe o ciclo quando o tempo estabelecido acaba
            selected_node = self.select(self.root)    # seleciona o nó folha a ser estudado nessa iteração
            if selected_node.visits == 0:
                result = self.rollout(selected_node)              # simula, a partir do nó atual, uma partida até chegar a um vencedor e guarda o número de quem venceu
                self.back_propagation(selected_node, result)
            else:
                selected_node.add_children() 
                for child in selected_node.select_children():
                    result = self.rollout(child[0])              # simula, a partir do nó atual, uma partida até chegar a um vencedor e guarda o número de quem venceu
                    self.back_propagation(child[0], result)
        return self.best_move()    


    def select(self, node: Node) ->  Node:
        """select a leaf node to be expanded/simulated"""
        if node.children == []: 
            return node
        else: 
            node = self.best_child(node)   # se o nó tiver filhos, seleciona o seu melhor filho; se não, retorna ele próprio
            return self.select(node)

    
    def best_child(self, node: Node) -> Node:
        """select the best child to be expanded based on their ucb's"""
        best_child = None
        best_score = float('-inf')
        for tuplo in node.children:
            child = tuplo[0]
            ucb = child.ucb() if child.visits != 0 else float("+inf")
            if ucb > best_score:
                best_child = child
                best_score = ucb
        return best_child


    def back_propagation(self, node: Node, result: int) -> None:
        """go through the tree to update the score of each node above the current one"""
        while node:    # itera sobre todos os nós "pais" do último nó da simulação
            node.visits += 1   # anota que mais uma simulação foi feita sobre esse nó
            if node.current_player == result:    # se o jogador desse nó tiver ganhado a partida simulada, anota mais uma vitória
                node.wins+=1           
            # self.update_best_node(node)
            node = node.parent              # passa ao pai do nó atual, para atualizar também o seu score
    

    def expand(self, node: Node) -> Node:
        """expand the node, by adding its children to the tree, and select one random child to be EXPLORED ?????????????????????????????W"""
        node.add_children()
        return random.choice(node.children)[0]   # seleciona aleatoriamente um dos filhos gerados e retorna seu tabuleiro
    
        
    def rollout(self, node: Node) -> int:
        """simulate a entire play until someone wins"""
        board = node.board.copy()   # cria uma cópia do tabuleiro do nó para ser alterado
        players = itertools.cycle([c.AI_PIECE, c.HUMAN_PIECE])  # cria uma iteração sobre os jogadores de cada nível
        current_player = next(players)
        while not (game.winning_move(board, c.AI_PIECE) or game.winning_move(board, c.HUMAN_PIECE)):   # continua a simulação até o jogo simulado acabar
            if game.is_game_tied(board): return 0
            current_player = next(players)
            values = game.available_moves(board)   # seleciona as colunas disponpiveis a receberem jogadas
            if values == -1:       # se não houver possibilidades de jogadas, retorna que não houve ganhador (empate)
                current_player = 0
                break
            col = random.choice(values)    # escolhe aleatoriamente uma das possibilidades de jogada
            board = game.simulate_move(board, current_player, col)  # simula a jogada escolhida
        return current_player       # retorna o jogador da jogada vencedora (ou seja, quem ganhou a simulação)
    

    def best_move(self) -> int:
        """select the best column to be played based on their scores"""
        max_uct = float('-inf')
        scores = {}    # ?????????????????????????????????????????????????
        columns = []   # armazena as colunas que têm o melhor score de vitórias
        print(self.root.children)
        for (child, col) in self.root.children:   # para cada possível jogada...
            uct = child.score()      # calcula o score do filho
            print(f"Coluna: {col}")
            print(child)
            if uct > max_uct:        
                max_uct = uct        # se esse for o novo melhor score, armazena como mehor
            scores[col] = uct        # ?????????????????????????????????????????????????????????????
        for col, score in scores.items(): 
            if score == max_uct:
                columns.append(col)    # seleciona todos os filhos que geram o melhor score
        return random.choice(columns)    # escolhe aleatoriamente um dos filhos com o melhor score, caso haja mais de um com o mesmo
            
def mcts(board: np.ndarray) -> int:
    """Should return the best column option, chose by mcts"""
    root = Node(board=board, last_player=c.AI_PIECE)
    mcts = MCTS(root)
    column = mcts.start(3)
    print(column+1)
    return column