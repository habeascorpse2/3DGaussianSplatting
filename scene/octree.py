import numpy as np

class Octree:
    def __init__(self, points, scales, rotations, min_cell_size=0.6):
        """
        Inicializa uma Octree dinâmica, que se subdivide até que o tamanho de cada célula seja >= min_cell_size.
        
        Cada ponto é considerado com um "bounding box" definido por p ± default_extent,
        onde default_extent é definido como um quarto do tamanho da célula (assumindo células cúbicas).
        Se o bounding box de um ponto se estender por mais de uma célula, o ponto é inserido em todas elas.
        
        :param points: numpy array de shape (N, 3) com as posições dos pontos.
        :param scales: numpy array de shape (N, 3) com as escalas de cada ponto.
        :param rotations: numpy array de shape (N, 3, 3) com as matrizes de rotação de cada ponto.
        :param min_cell_size: tamanho mínimo permitido para uma célula.
        """
        self.points = np.asarray(points)      # shape (N, 3)
        self.scales = np.asarray(scales)      # shape (N, 3)
        self.rotations = np.asarray(rotations)  # shape (N, 3, 3)
        self.min_cell_size = min_cell_size
        
        # Calcula o bounding box básico dos pontos
        basic_min = self.points.min(axis=0)
        basic_max = self.points.max(axis=0)
        center = (basic_min + basic_max) / 2.0
        # Para garantir células cúbicas, usa o maior alcance entre os eixos
        range_val = (basic_max - basic_min).max()
        # Define o bounding box global como um cubo centrado no centro dos pontos
        self.global_min = center - range_val / 2.0
        self.global_max = center + range_val / 2.0
        
        # Determina a profundidade máxima para que o tamanho da célula não seja menor que min_cell_size.
        # A cada subdivisão, o tamanho da célula é: cell_size = range_val / (2**depth)
        self.depth = int(np.floor(np.log2(range_val / min_cell_size))) if range_val > min_cell_size else 0
        self.n_cells = 2 ** self.depth if self.depth > 0 else 1
        # Como o bounding box é cúbico, o tamanho da célula é o mesmo em todas as direções.
        self.cell_size = (self.global_max - self.global_min) / self.n_cells
        
        # Define a extensão padrão para cada ponto: um quarto do tamanho da célula (assumindo células uniformes).
        # Como self.cell_size é um vetor (igual em todas as direções), usamos o primeiro componente.
        self.default_extent = self.cell_size[0] / 4.0
        
        # Dicionário para armazenar, para cada célula (índice tuple), os índices dos pontos que nela se encontram.
        self.cells = {}
        # Dicionário para mapear cada índice de ponto para a lista de células (índices) onde ele está inserido.
        self.point_cells = {}
        
        # Insere todos os pontos na árvore
        N = self.points.shape[0]
        for i in range(N):
            self._insert_point(i, self.points[i], self.scales[i], self.rotations[i])
    
    def _get_cell_indices_for_point(self, point):
        """
        Calcula os índices das células onde o ponto, considerando seu bounding box (p ± default_extent),
        intersecciona o bounding box global.
        """
        p_min = point - self.default_extent
        p_max = point + self.default_extent
        start_idx = np.floor((p_min - self.global_min) / self.cell_size).astype(int)
        end_idx = np.floor((p_max - self.global_min) / self.cell_size).astype(int)
        start_idx = np.clip(start_idx, 0, self.n_cells - 1)
        end_idx = np.clip(end_idx, 0, self.n_cells - 1)
        
        indices = []
        for ix in range(start_idx[0], end_idx[0] + 1):
            for iy in range(start_idx[1], end_idx[1] + 1):
                for iz in range(start_idx[2], end_idx[2] + 1):
                    indices.append((ix, iy, iz))
        return indices

    def _insert_point(self, index, point, scale, rotation):
        """
        Insere o ponto de índice 'index' na árvore, registrando em quais células ele se encontra.
        """
        cell_indices = self._get_cell_indices_for_point(point)
        self.point_cells[index] = cell_indices
        for cell in cell_indices:
            if cell not in self.cells:
                self.cells[cell] = []
            self.cells[cell].append(index)
    
    def update_point(self, index, new_point, new_scale, new_rotation):
        """
        Atualiza os dados do ponto de índice 'index'. Remove-o das células antigas,
        atualiza a posição, escala e rotação, e o reinsere na octree.
        """
        self.remove_point(index)
        self.points[index] = new_point
        self.scales[index] = new_scale
        self.rotations[index] = new_rotation
        self._insert_point(index, new_point, new_scale, new_rotation)
    
    def remove_point(self, index):
        """
        Remove o ponto de índice 'index' da octree, retirando-o de todas as células em que foi inserido.
        """
        if index in self.point_cells:
            for cell in self.point_cells[index]:
                if cell in self.cells and index in self.cells[cell]:
                    self.cells[cell].remove(index)
            del self.point_cells[index]
    
    def insert_point(self, point, scale, rotation):
        """
        Insere um novo ponto na octree e retorna o índice atribuído a ele.
        """
        index = self.points.shape[0]
        self.points = np.vstack((self.points, point.reshape(1, 3)))
        self.scales = np.vstack((self.scales, scale.reshape(1, 3)))
        self.rotations = np.concatenate((self.rotations, rotation.reshape(1, 3, 3)), axis=0)
        self._insert_point(index, point, scale, rotation)
        return index
    
    def query_cell(self, point):
        """
        Retorna o índice da célula (tupla de 3 inteiros) na qual o ponto se encontra.
        """
        point = np.asarray(point)
        rel_pos = (point - self.global_min) / (self.global_max - self.global_min)
        indices = np.floor(rel_pos * self.n_cells).astype(int)
        indices = np.clip(indices, 0, self.n_cells - 1)
        return tuple(indices)
    
    def query(self, point):
        """
        Retorna a lista de índices dos pontos armazenados na célula onde o ponto se encontra.
        """
        cell_index = self.query_cell(point)
        return self.cells.get(cell_index, [])