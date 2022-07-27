#include <bits/stdc++.h>

using namespace std;

vector<vector<int>> read_csv(string filename) //传入文件所在的位置
{
    ifstream inFile(filename);
    string lineStr;
    vector<vector<int>> strArray; //用来保存读取出来的数据，可以看成是一个二维数组，类型一般是string，其他类型可以转换
    cout << "the whole line is: " << endl;
    while (getline(inFile, lineStr)) //这里的循环是每次读取一整行的数据,把结果保存在lineStr中，lineStr是用逗号分割开的
    {
        //打印整行字符串
        // cout<<lineStr<<endl;
        //将结果保存为二维表结构
        stringstream ss(lineStr); //这里stringstream是一个字符串流类型，用lineStr来初始化变量 ss
        string str;
        vector<int> lineArray;
        //按照逗号进行分割
        while (getline(ss, str, ',')) // getline每次把按照逗号分割之后的每一个字符串都保存在str中
        {
            lineArray.push_back(atoi(str.c_str())); //这里将str保存在lineArray中
        }
        strArray.push_back(lineArray); //这里把lineArray保存在strArray。   这里的lineArray和lineArray有点类似于python中的list，只是固定了要保存的数据类型
    }
    cout << "--------------------------------------------" << endl;
    //打印一下我们保存的数据
    cout << " vector loaded " << endl;
    return strArray;
}

int main()
{
    //读取整个文件
    string data_dir = "../datasets/NA12878_PacBio_MtSinai/";
    string filename = data_dir + "chromosome_sign/" + chromosome + "_m(i)d_sign.csv";
    vector<vector<int>> data = read_csv(filename);
    return 0;
}
