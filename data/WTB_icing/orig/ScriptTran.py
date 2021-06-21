import csv

with open('./21/21_normalInfo_raw.csv', 'r') as read:
    reader = csv.reader(read)

    with open('./21/21_normalInfo.csv', 'w') as write:
        writer = csv.writer(write)
        writer.writerow(next(reader))
        for read_row in reader:
            read_row[0] = read_row[0] + ':00'
            read_row[1] = read_row[1] + ':00'
            writer.writerow(read_row)