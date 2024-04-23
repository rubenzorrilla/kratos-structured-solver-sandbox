import scipy.io

print("Reading pressure_matrix.mm...")
pressure_matrix = scipy.io.mmread('pressure_matrix.mm')
print("pressure_matrix.mm imported.")

print("Reading pressure_matrix_periodic.mm...")
pressure_matrix_periodic = scipy.io.mmread('pressure_matrix_periodic.mm')
print("pressure_matrix_periodic.mm imported.")

print(pressure_matrix)
print(pressure_matrix_periodic)
