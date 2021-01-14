


def parse_result_table2csv(result_table,save_path):
    csv_list = []
    csv_list.append("{},{},{},{},{},{},{}".format("label","annots","dets","hits","ap","precision","recall"))
    with open(result_table,'r') as f:
        for one_line in f:
            one_list = one_line.strip().split("\t")
            label = one_list[0]
            annot = one_list[1].strip().split(":")[-1]
            dets = one_list[2].strip().split(":")[-1]
            hit = one_list[3].strip().split(":")[-1]
            ap = one_list[5].strip().split(":")[-1]
            precision = one_list[6].strip().split(":")[-1]
            recall = one_list[7].strip().split(":")[-1]
            csv_list.append("{},{},{},{},{},{},{}".format(label,annot,dets,hit,ap,precision,recall))
    print(csv_list)
    with open(save_path,'w') as f:
        f.write('\n'.join(csv_list))


if __name__ == '__main__':
    result_table = r'E:\codeOfMe\stafen_material\项目-模型-结果\模型测试日志\2019-12-18\sign-V187\13-classes-result-table-20191221-conf0.6.txt'
    save_path = r'E:\codeOfMe\stafen_material\项目-模型-结果\模型测试日志\2019-12-18\sign-V187\13-classes-result-table-20191221-conf0.6.csv'
    parse_result_table2csv(result_table,save_path)