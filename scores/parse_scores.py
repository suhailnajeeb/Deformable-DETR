from prettytable import PrettyTable

output_paths = [
    'eval_iter_2s_64.out',
    'eval_iter_2s_69.out',
    'eval_iter_2s_74.out',
    'eval_iter_2s_79.out',
    'eval_iter_2s_84.out',
    'eval_iter_2s_89.out',
    'eval_iter_2s_94.out',
    'eval_iter_2s_ckp.out',
]

for output_path in output_paths:
    #output_path = 'eval_iter_2s_ckp.out'
    table_name = output_path.split('.')[0] + '.csv'

    with open(output_path, 'r') as f:
        output = f.readlines()

    table = PrettyTable(['Country', 'mAP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100'])

    for line in output:
        if "Processing data for:" in line: 
            country = line.split(': ')[1][:-1]
        if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]' in line:
            mAP = line[-6:-1]
        if 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]' in line:
            AP50 = line[-6:-1]
        if 'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]' in line:
            AP75 = line[-6:-1]
        if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]' in line:
            APs = line[-6:-1]
        if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]' in line:
            APm = line[-6:-1]
        if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]' in line:
            APl = line[-6:-1]
        if 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]' in line:
            AR1 = line[-6:-1]
        if 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]' in line:
            AR10 = line[-6:-1]
        if 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]' in line:
            AR100 = line[-6:-1]
            table.add_row([country, mAP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100])

    print('Scores for file: {}'.format(output_path))
    print(table)

    # Save th table to a CSV file: 
    with open(table_name, 'w') as f: 
        f.write(table.get_csv_string())