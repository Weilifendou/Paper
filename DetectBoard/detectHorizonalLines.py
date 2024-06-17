
def detectHorizontalLines(edges, length):
    lines = []
    flag = 0
    row, col = edges.shape
    for r in range(row):
        for c in range(col):
            if edges[r, c] == 255:
                if flag == 0:
                    startX = c
                    startY = r
                    flag = 1
                else:
                    continue
            else:
                if flag == 1:
                    endX = c
                    endY = r
                    if endX - startX > length:
                        lines.append((startX, startY, endX, endY))
                    flag = 0
    return lines