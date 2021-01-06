""" Modelo para resolver un problema de mochila sin algoritmo genético. Se probó
el comportamiento de las instancias en un primer momento"""

def knapsack(peso_permitido, objetos):  #codigo basado en https://github.com/satuelisa/Simulation/blob/master/GeneticAlgorithm/knapsack.py
    peso_total = sum([objeto[0] for objeto in objetos])
    valor_total = sum([objeto[1] for objeto in objetos])
    if peso_total < peso_permitido: # cabe todo
        return valor_total
    else:
        cantidad = len(objetos)
        V = dict()
        for w in range(peso_permitido + 1):
            V[(w, 0)] = 0
        for i in range(cantidad):
            (peso, valor) = objetos[i]
            for w in range(peso_permitido + 1):
                cand = V.get((w - peso, i), -float('inf')) + valor
                V[(w, i + 1)] = max(V[(w, i)], cand)
        return max(V.values())
 
peso = [15, 23, 28, 29, 30, 33, 34, 35, 36, 40, 45, 47, 48, 49, 51, 53, 58, 59, 60, 61, 69, 75, 76, 78, 79, 80, 82, 99] # ordenados de menor a mayor
valor = [630, 591, 630, 640, 869, 932, 934, 504, 803, 667, 1103, 834, 585, 811, 856, 690, 832, 846, 813, 868, 793, 825, 1002, 860, 615, 540, 797, 616]
objetos = [(p, v) for (p, v) in zip(peso, valor)]
print(knapsack(100, objetos))