#
# 2020适合买房吗？
# 袖满飞花
# 行走世间，都是妖怪。 没有IDE就不会编程星人。
#
# 我来从不同的角度提供一个思路。
#
# 我希望计算出买房和不买房的两种情况下，在未来某一年我的所有的资产的价值。如果买房我的所有资产价值高，那么就应该买房。如果不买房我的所有资产的价格高，那么就不应该买房。
#
# 计算时需要初始的参数，以下是我计算时使用的初始参数。
#
# 假定：
#
# 我现在手头的现金有300w，我希望买一个现在标价420w的房子，中介费是12w。
#
# 我的还款方式是等额本金（等额本金易于计算），贷款年限为15年，首付250w。
#
# 房贷利息按年算，是5%。我的工资收入是40w，并且假定在未来十年内工资收入不变。
#
# 银行的理财利息是3.8%（招商银行理财产品目前利息），并且假定理财利息也是十年内不变。
#
# 如果不买房的话，房租是一年6.5w元。
#
# 如果买房，十年后，我买的房子市值从420w涨到了550w。
#
# 那么如果我买房，我十年后的总资产大于我不买房的总资产，那么此时我应该买房。反之，如果我买房十年后的总资产小于我不买房的总资产，那么我此时就不应该买房。
#
#
# 如果买房，那么我十年后的总资产是：（当前手头现金-中介费-房屋首付）* 理财利息 + （当前房屋市值 - 剩余欠银行的房贷）+ （十年间工资还完房贷部分的理财收入）
#
# 如果不买房，那么我十年后的总资产是（当前手头现金） * 十年的理财利息 + （十年间工资付清房租部分的理财收入）
#
# 注意：因为是否买房，都要有吃喝拉撒睡等日常开支，我们假定这部分开支是一样的，因此不予考虑。
#
# 以下是我计算的python代码，欢迎大家review，同时也欢迎根据实际情况完善模型。
#

import math

OriMoney = 3000000  # 现有的所有现金
AgentFee = 120000  # 现在买房付的中介费
HouseOriPrice = 4200000  # 所买房屋价格
EvalYear = 10  # 十年后评估
Inflation = 1.038  # 理财利息
HouseCurPrice = 5500000  # 假定十年后房屋的价格
DownPayment = 2500000  # 现在买房所需首付
LoanYear = 15  # 总计贷款年限
YearSalary = 400000  # 每年的工资收入
LoanRatio = 0.05  # 房贷利息
HouseRentYear = 65000  # 每年所付的房租


def main():
    BuyHouseProperty = getBuyHouseProperty()
    print('cur Property if buy a house:\t' + str(BuyHouseProperty))
    NoHouseProperty = getNoHouseProperty()
    print('cur Property if not buy the house:\t' + str(NoHouseProperty))
    return


def getBuyHouseProperty():
    # 现有现金去除买房款后的理财收入（复利）
    moneyGain = (OriMoney - AgentFee - DownPayment) * math.pow(Inflation, EvalYear)
    # 现有现金去除买房款后的理财收入（复利，同上述写法）
    moneyGain = (OriMoney - AgentFee - DownPayment) * (Inflation ** EvalYear)
    # 买房后产生的资产增值
    HouseGain = (HouseCurPrice - (HouseOriPrice - DownPayment) / float(LoanYear) * (LoanYear - EvalYear))
    # 每年工资收入减去房贷后的净收入
    SalaryGain = getSalaryGain()

    return moneyGain + HouseGain + SalaryGain


def getSalaryGain():
    SalaryGain = 0
    for i in range(EvalYear):
        n = i + 1
        # 第n年需要付的房贷
        YearPay = (HouseOriPrice - DownPayment) * float(LoanYear * LoanRatio - n * LoanRatio + LoanRatio + 1) / LoanYear
        # 从第n年到最后一年，第n年工资进行理财得到的收入
        ThisYearSalary = (YearSalary - YearPay) * math.pow(Inflation, EvalYear - n)
        SalaryGain += ThisYearSalary

    return SalaryGain


def getNoHouseProperty():
    # 现有现金的理财收入
    MoneyGain = OriMoney * math.pow(Inflation, EvalYear)
    SalaryGain = 0
    for i in range(EvalYear):
        n = i + 1
        # 第n年的工资去除开销后的理财收入
        ThisYearSalary = (YearSalary - HouseRentYear) * math.pow(Inflation, EvalYear - n)
        SalaryGain += ThisYearSalary

    return MoneyGain + SalaryGain


if __name__ == "__main__":
    main()

#
# 以上代码我跑出来的结果是：
#
# cur Property if buy a house: 8166636.06869
#
# cur Property if not buy the house: 8341010.16549
#
# 也就是说呢，如果我不买房，十年后，我的所有资产折现是834w，如果我买房呢，十年后我的所有资产折现是816.6w。
#
# 把你所有的假设参数带进去，算一算，就知道，2020年到底适不适合买房啦。
#
# 当然，如果你觉得现在420w的房子，十年后值1000w。
#
# 那老铁，等啥呢？看啥知乎啊，赶紧买呀！
#
# 如果现在420w的房子，十年后你觉得值500w。
#
# 那老铁，要不你再等等？
#
# 也可以用这个方法计算一下。假如18年买了一个价值500w的房子，两年房价不涨，那现在浮亏50w。
