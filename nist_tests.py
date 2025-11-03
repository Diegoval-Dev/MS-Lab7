import numpy as np
from scipy import stats
from scipy.stats import chi2
from lcg_generator import LCG
from mersenne_twister import generar_muestra_mt
import math


# ============================================================
# Funciones para generar secuencias de bits
# ============================================================

def generar_bits_lcg(n_bits: int, seed: int = 12345, 
                     a: int = 1664525, c: int = 1013904223, 
                     m: int = 2**32) -> np.ndarray:
    """
    Genera una secuencia de n_bits bits usando el generador LCG.
    Convierte los números aleatorios uniformes en bits.
    """
    lcg = LCG(seed=seed, a=a, c=c, m=m)
    bits = []
    
    # Generar números aleatorios y convertir a bits
    # Método: usar múltiples bits de cada número para mayor aleatoriedad
    n_numeros = (n_bits // 32) + 1
    
    for _ in range(n_numeros):
        num = lcg.next()
        # Convertir a bits (usar los 32 bits del número)
        for i in range(32):
            bit = (num >> i) & 1
            bits.append(bit)
            if len(bits) >= n_bits:
                break
        if len(bits) >= n_bits:
            break
    
    return np.array(bits[:n_bits], dtype=int)


def generar_bits_mt(n_bits: int, seed: int = 42) -> np.ndarray:
    """
    Genera una secuencia de n_bits bits usando el generador Mersenne Twister.
    """
    rng = np.random.default_rng(seed)
    n_numeros = (n_bits // 32) + 1
    
    bits = []
    for _ in range(n_numeros):
        num = rng.integers(0, 2**32, dtype=np.uint32)
        # Convertir a bits
        for i in range(32):
            bit = (num >> i) & 1
            bits.append(bit)
            if len(bits) >= n_bits:
                break
        if len(bits) >= n_bits:
            break
    
    return np.array(bits[:n_bits], dtype=int)


# ============================================================
# Implementación de pruebas NIST SP 800-22
# ============================================================

def test_frequency(bits: np.ndarray):
    """
    Prueba de Frecuencia (Monobit Test)
    Evalúa si el número de unos y ceros en la secuencia es aproximadamente igual.
    """
    n = len(bits)
    suma = np.sum(2 * bits - 1)  # Convertir 0->-1, 1->1
    s_obs = abs(suma) / math.sqrt(n)
    p_value = math.erfc(s_obs / math.sqrt(2))
    return p_value


def test_block_frequency(bits: np.ndarray, block_size: int = 128):
    """
    Prueba de Frecuencia en Bloques
    Evalúa la frecuencia de unos en bloques de M bits.
    """
    n = len(bits)
    num_blocks = n // block_size
    if num_blocks < 1:
        return None
    
    blocks = bits[:num_blocks * block_size].reshape(num_blocks, block_size)
    proportions = np.sum(blocks, axis=1) / block_size
    
    chi_square = 4 * block_size * np.sum((proportions - 0.5)**2)
    p_value = 1 - chi2.cdf(chi_square, num_blocks)
    return p_value


def test_runs(bits: np.ndarray):
    """
    Prueba de Corridas (Runs Test)
    Evalúa el número total de corridas en la secuencia.
    """
    n = len(bits)
    pi = np.sum(bits) / n
    
    if abs(pi - 0.5) >= (2.0 / math.sqrt(n)):
        return 0.0  # No pasa la prueba de frecuencia previa
    
    # Contar corridas
    runs = 1
    for i in range(1, n):
        if bits[i] != bits[i-1]:
            runs += 1
    
    numerador = abs(runs - 2 * n * pi * (1 - pi))
    denominador = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    
    if denominador == 0:
        return None
    
    z = numerador / denominador
    p_value = math.erfc(z)
    return p_value


def test_longest_run_of_ones(bits: np.ndarray):
    """
    Prueba de la Corrida Más Larga de Unos
    Evalúa la corrida más larga de unos dentro de bloques de M bits.
    """
    n = len(bits)
    
    # Determinar tamaño de bloque según n
    if n < 6272:
        M = 8
        K = 3
        pi = [0.2148, 0.3672, 0.2305, 0.1875]  # K+2 = 5, pero usamos 4 valores + categoría >K
    elif n < 750000:
        M = 128
        K = 5
        pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]  # K+2 = 7, tenemos 6 valores
    else:
        M = 10000
        K = 6
        pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]  # K+2 = 8, tenemos 7 valores
    
    num_blocks = n // M
    if num_blocks < 1:
        return None
    
    # Encontrar la corrida más larga en cada bloque
    blocks = bits[:num_blocks * M].reshape(num_blocks, M)
    longest_runs = []
    
    for block in blocks:
        max_run = 0
        current_run = 0
        for bit in block:
            if bit == 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        longest_runs.append(min(max_run, K + 1))
    
    # Contar frecuencias
    num_categories = len(pi)
    freq = np.zeros(num_categories)
    for run in longest_runs:
        if run < num_categories - 1:
            freq[int(run)] += 1
        else:
            freq[num_categories - 1] += 1  # Categoría para runs >= num_categories-1
    
    # Calcular chi-cuadrado
    chi_square = 0
    for i in range(num_categories):
        expected = num_blocks * pi[i]
        if expected > 0:
            chi_square += (freq[i] - expected)**2 / expected
    
    p_value = 1 - chi2.cdf(chi_square, num_categories - 1)
    return p_value


def test_binary_matrix_rank(bits: np.ndarray):
    """
    Prueba de Rango de Matriz Binaria
    Evalúa el rango de matrices binarias formadas por la secuencia.
    """
    n = len(bits)
    M = Q = 32  # Tamaño de matriz
    
    num_matrices = n // (M * Q)
    if num_matrices < 1:
        return None
    
    # Contar matrices de rango M, M-1, y < M-1
    ranks = np.zeros(3)
    
    for i in range(num_matrices):
        start = i * M * Q
        matrix = bits[start:start + M * Q].reshape(M, Q)
        # Calcular rango (simplificado)
        rank = np.linalg.matrix_rank(matrix.astype(float))
        if rank == M:
            ranks[0] += 1
        elif rank == M - 1:
            ranks[1] += 1
        else:
            ranks[2] += 1
    
    # Calcular p-value usando distribución chi-cuadrado
    N = num_matrices
    expected = np.array([
        N * 0.2888,  # P(rango = M)
        N * 0.5776,  # P(rango = M-1)
        N * 0.1336   # P(rango < M-1)
    ])
    
    chi_square = np.sum((ranks - expected)**2 / expected)
    p_value = 1 - chi2.cdf(chi_square, 2)
    return p_value


def test_discrete_fourier_transform(bits: np.ndarray):
    """
    Prueba de Transformada Discreta de Fourier
    Evalúa los picos en la transformada de Fourier de la secuencia.
    """
    n = len(bits)
    # Convertir bits a valores -1 y 1
    sequence = 2 * bits - 1
    
    # Calcular DFT
    fft_result = np.fft.fft(sequence)
    modulus = np.abs(fft_result[:n//2])
    
    # Contar picos que están por debajo del umbral
    threshold = math.sqrt(math.log(1.0 / 0.05) * n)
    N0 = 0.95 * n / 2.0
    N1 = np.sum(modulus < threshold)
    
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4)
    p_value = math.erfc(abs(d) / math.sqrt(2))
    return p_value


def test_cumulative_sums(bits: np.ndarray, mode: str = 'forward'):
    """
    Prueba de Sumas Acumulativas
    Evalúa las sumas acumulativas parciales de la secuencia.
    """
    n = len(bits)
    # Convertir bits a valores -1 y 1
    sequence = 2 * bits - 1
    
    if mode == 'forward':
        s = np.cumsum(sequence)
    else:  # backward
        s = np.cumsum(sequence[::-1])
    
    z = np.max(np.abs(s))
    
    # Calcular p-value
    suma = 0
    for k in range(int((-n / z) + 1), int((n / z) - 1) + 1):
        suma += (1 - 4 * k * z / n) * math.exp(-2 * (k * z / n)**2)
    
    p_value = suma
    return p_value


def test_serial(bits: np.ndarray, pattern_length: int = 16):
    """
    Prueba de Serial
    Evalúa la frecuencia de patrones de m bits.
    """
    n = len(bits)
    m = min(pattern_length, int(math.log2(n)) - 2)
    if m < 1:
        return None
    
    # Contar frecuencias de patrones de m y m-1 bits
    def count_patterns(bits_array, m_pattern):
        counts = {}
        for i in range(len(bits_array) - m_pattern + 1):
            pattern = tuple(bits_array[i:i+m_pattern])
            counts[pattern] = counts.get(pattern, 0) + 1
        return counts
    
    psi_m = count_patterns(bits, m)
    psi_m_minus_1 = count_patterns(bits, m - 1)
    
    # Calcular psi2
    sum_squared = sum(v**2 for v in psi_m.values())
    sum_m_minus_1_squared = sum(v**2 for v in psi_m_minus_1.values())
    
    psi2_m = (2**m / n) * sum_squared - n
    psi2_m_minus_1 = (2**(m-1) / n) * sum_m_minus_1_squared - n
    
    delta_psi_sq = psi2_m - psi2_m_minus_1
    p_value1 = 1 - chi2.cdf(delta_psi_sq, 2**(m-2))
    p_value2 = 1 - chi2.cdf(psi2_m, 2**(m-1))
    
    return min(p_value1, p_value2)


def test_approximate_entropy(bits: np.ndarray, m: int = 10):
    """
    Prueba de Entropía Aproximada
    Evalúa la frecuencia de patrones de m y m+1 bits.
    """
    n = len(bits)
    m = min(m, int(math.log2(n)) - 1)
    if m < 1:
        return None
    
    def phi(bits_array, m_pattern):
        counts = {}
        for i in range(len(bits_array) - m_pattern + 1):
            pattern = tuple(bits_array[i:i+m_pattern])
            counts[pattern] = counts.get(pattern, 0) + 1
        
        N = len(bits_array) - m_pattern + 1
        suma = 0
        for count in counts.values():
            if count > 0:
                p = count / N
                suma += p * math.log(p)
        return suma
    
    phi_m = phi(bits, m)
    phi_m_plus_1 = phi(bits, m + 1)
    
    ap_en = phi_m - phi_m_plus_1
    
    # Calcular p-value
    chi_square = 2 * n * (math.log(2) - ap_en)
    p_value = 1 - chi2.cdf(chi_square, 2**m)
    return p_value


# ============================================================
# Función para ejecutar todas las pruebas
# ============================================================

def ejecutar_pruebas_nist(bits: np.ndarray, nombre_generador: str):
    """
    Ejecuta todas las pruebas NIST SP 800-22 en la secuencia de bits.
    Retorna un diccionario con los resultados.
    """
    resultados = {}
    
    print(f"Ejecutando pruebas para {nombre_generador}...")
    
    pruebas = [
        ("Frecuencia (Monobit)", test_frequency),
        ("Frecuencia en Bloques", lambda b: test_block_frequency(b, 128)),
        ("Corridas (Runs)", test_runs),
        ("Corrida Más Larga de Unos", test_longest_run_of_ones),
        ("Rango de Matriz Binaria", test_binary_matrix_rank),
        ("Transformada Discreta de Fourier", test_discrete_fourier_transform),
        ("Sumas Acumulativas (Adelante)", lambda b: test_cumulative_sums(b, 'forward')),
        ("Sumas Acumulativas (Atrás)", lambda b: test_cumulative_sums(b, 'backward')),
        ("Serial", lambda b: test_serial(b, 16)),
        ("Entropía Aproximada", lambda b: test_approximate_entropy(b, 10)),
    ]
    
    for nombre, funcion in pruebas:
        try:
            p_value = funcion(bits)
            if p_value is not None:
                passed = p_value >= 0.01
                resultados[nombre] = {
                    'p_value': p_value,
                    'passed': passed,
                    'resultado': 'PASÓ' if passed else 'FALLÓ'
                }
            else:
                resultados[nombre] = {
                    'p_value': None,
                    'passed': False,
                    'resultado': 'N/A'
                }
        except Exception as e:
            print(f"Error en {nombre}: {e}")
            resultados[nombre] = {
                'p_value': None,
                'passed': False,
                'resultado': 'ERROR'
            }
    
    return resultados


# ============================================================
# Función principal
# ============================================================

def main():
    print("=" * 80)
    print("PRUEBAS NIST SP 800-22 - EVALUACIÓN DE GENERADORES PSEUDOALEATORIOS")
    print("=" * 80)
    print()
    
    N_BITS = 1_000_000
    print(f"Generando {N_BITS:,} bits para cada generador...")
    print()
    
    # Generar bits con LCG
    print("Generando secuencia de bits con LCG...")
    bits_lcg = generar_bits_lcg(N_BITS, seed=12345)
    print(f"Bits generados: {len(bits_lcg):,}")
    print(f"Proporción de 1s: {np.mean(bits_lcg):.4f}")
    print()
    
    # Generar bits con Mersenne Twister
    print("Generando secuencia de bits con Mersenne Twister...")
    bits_mt = generar_bits_mt(N_BITS, seed=42)
    print(f"Bits generados: {len(bits_mt):,}")
    print(f"Proporción de 1s: {np.mean(bits_mt):.4f}")
    print()
    
    # Ejecutar pruebas NIST para LCG
    print("=" * 80)
    print("Ejecutando pruebas NIST SP 800-22 para LCG...")
    print("=" * 80)
    resultados_lcg = ejecutar_pruebas_nist(bits_lcg, "LCG")
    print()
    
    # Ejecutar pruebas NIST para Mersenne Twister
    print("=" * 80)
    print("Ejecutando pruebas NIST SP 800-22 para Mersenne Twister...")
    print("=" * 80)
    resultados_mt = ejecutar_pruebas_nist(bits_mt, "Mersenne Twister")
    print()
    
    # Crear tabla comparativa
    print()
    print("=" * 80)
    print("TABLA COMPARATIVA DE RESULTADOS")
    print("=" * 80)
    print()
    
    # Crear lista de datos para la tabla
    tabla_datos = []
    todas_las_pruebas = sorted(set(list(resultados_lcg.keys()) + list(resultados_mt.keys())))
    
    for prueba in todas_las_pruebas:
        res_lcg = resultados_lcg.get(prueba, {})
        res_mt = resultados_mt.get(prueba, {})
        
        p_lcg = res_lcg.get('p_value')
        p_mt = res_mt.get('p_value')
        
        tabla_datos.append({
            'Prueba': prueba,
            'LCG p-value': f"{p_lcg:.6f}" if p_lcg is not None else "N/A",
            'LCG Resultado': res_lcg.get('resultado', 'N/A'),
            'MT p-value': f"{p_mt:.6f}" if p_mt is not None else "N/A",
            'MT Resultado': res_mt.get('resultado', 'N/A')
        })
    
    # Mostrar tabla formateada
    print(f"{'Prueba':<50} {'LCG p-value':<15} {'LCG Resultado':<15} {'MT p-value':<15} {'MT Resultado':<15}")
    print("=" * 110)
    for fila in tabla_datos:
        print(f"{fila['Prueba']:<50} {fila['LCG p-value']:<15} {fila['LCG Resultado']:<15} "
              f"{fila['MT p-value']:<15} {fila['MT Resultado']:<15}")
    print()
    
    # Estadísticas resumidas
    print("=" * 80)
    print("RESUMEN ESTADÍSTICO")
    print("=" * 80)
    
    pruebas_pasadas_lcg = sum(1 for r in resultados_lcg.values() 
                              if r.get('p_value') is not None and r.get('passed', False))
    pruebas_totales_lcg = sum(1 for r in resultados_lcg.values() 
                              if r.get('p_value') is not None)
    pruebas_pasadas_mt = sum(1 for r in resultados_mt.values() 
                            if r.get('p_value') is not None and r.get('passed', False))
    pruebas_totales_mt = sum(1 for r in resultados_mt.values() 
                            if r.get('p_value') is not None)
    
    print(f"LCG: {pruebas_pasadas_lcg}/{pruebas_totales_lcg} pruebas pasadas "
          f"({100*pruebas_pasadas_lcg/max(pruebas_totales_lcg,1):.1f}%)")
    print(f"Mersenne Twister: {pruebas_pasadas_mt}/{pruebas_totales_mt} pruebas pasadas "
          f"({100*pruebas_pasadas_mt/max(pruebas_totales_mt,1):.1f}%)")
    print()
    
    # Calcular p-values promedio
    p_values_lcg = [r['p_value'] for r in resultados_lcg.values() 
                    if r.get('p_value') is not None]
    p_values_mt = [r['p_value'] for r in resultados_mt.values() 
                  if r.get('p_value') is not None]
    
    if p_values_lcg:
        print(f"LCG - p-value promedio: {np.mean(p_values_lcg):.6f}")
        print(f"LCG - p-value mínimo: {np.min(p_values_lcg):.6f}")
    if p_values_mt:
        print(f"Mersenne Twister - p-value promedio: {np.mean(p_values_mt):.6f}")
        print(f"Mersenne Twister - p-value mínimo: {np.min(p_values_mt):.6f}")
    print()
    
    # Conclusión
    print("=" * 80)
    print("CONCLUSIÓN")
    print("=" * 80)
    
    if pruebas_totales_mt > 0 and pruebas_totales_lcg > 0:
        if pruebas_pasadas_mt > pruebas_pasadas_lcg:
            print("El generador Mersenne Twister muestra un mejor desempeño, ")
            print("pasando más pruebas estadísticas NIST SP 800-22 que el LCG.")
        elif pruebas_pasadas_lcg > pruebas_pasadas_mt:
            print("El generador LCG muestra un mejor desempeño, ")
            print("pasando más pruebas estadísticas NIST SP 800-22 que el Mersenne Twister.")
        else:
            print("Ambos generadores muestran un desempeño similar en las pruebas NIST SP 800-22.")
        
        if p_values_lcg and p_values_mt:
            if np.mean(p_values_mt) > np.mean(p_values_lcg):
                print("Además, el Mersenne Twister presenta p-values promedio más altos,")
                print("lo que indica una mejor calidad de aleatoriedad.")
            elif np.mean(p_values_lcg) > np.mean(p_values_mt):
                print("Además, el LCG presenta p-values promedio más altos,")
                print("lo que indica una mejor calidad de aleatoriedad.")
    
    print()
    print("Nota: Un p-value >= 0.01 generalmente indica que la secuencia pasa la prueba.")
    print("Un p-value < 0.01 sugiere que la secuencia no es suficientemente aleatoria.")


if __name__ == "__main__":
    main()