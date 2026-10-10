A = [[[1, 2, 3],
      [4, 5, 6],
      [3, 2, 1]],
     [[4, 5, 1],
      [1, 2, 2],
      [2, 2, 1]]]

B = [[[5, 3, 6],
      [6, 5, 4],
      [3, 2, 1]],
     [[1, 2, 3],
      [4, 5, 6],
      [1, 4, 3]]]

print("=== ARRAY A ===")
for k in range(2):
    print("Matriks", k + 1)
    for baris in A[k]:
        print(baris)
    print()

print("=== ARRAY B ===")
for k in range(2):
    print("Matriks", k + 1)
    for baris in B[k]:
        print(baris)
    print()

hasil = []

for k in range(2):
    matriks = []

    for i in range(3):
        baris = []

        for j in range(3):
            jumlah = 0

            for x in range(3):
                jumlah += A[k][i][x] * B[k][x][j]
            baris.append(jumlah)
        matriks.append(baris)
    hasil.append(matriks)

print("=== HASIL A x B ===")
for k in range(2):
    print("Matriks", k + 1)
    for baris in hasil[k]:
        print(baris)
    print()