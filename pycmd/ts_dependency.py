# internal
from .ast_util import findASTNode, srcToPos, getMaliciousChains, srcToFirstPos

CONSTRUCTOR_FLAG = "constructor"
FALLBACK_FLAG = "fallback"
UINT256_FLAG = "uint256"
ADD_EQU_FLAG = "+="
EQU_FLAG = "="
ADD_FLAG = "+"
SUB_EQU_FLAG = "-="
SUB_FLAG = "-"
SAFEMATH_FLAG = "SAFEMATH"
LIBRARY_FLAG = "library"
ADD_STR_FLAG = "add"
SUB_STR_FLAG = "sub"
DATASET_PATH = "./dataset/"
EXTRACTED_CONTRACT_SUFFIX = "_timestamp.sol"

FLAG_IF = -1
FLAG_FUNC = -2
FLAG_MODIFIER = -3
FLAG_VARDEC = -4
FLAG_ASSIGN = -5
FLAG_REQUIRE = -6

if __name__ == '__main__':
    pass

# find all block.timestamp and return a list of their location (start byte, end byte)
def getBlockTimeStamp(ast_json):
    '''
    :param ast_json: json of ast, loaded as a python dict
    :return: a list of locations of call.value (start byte, end byte)
    '''
    memAST = findASTNode(ast_json, 'name', 'MemberAccess')
    location = []
    for memItem in memAST:
        if memItem["attributes"]['member_name'] == 'timestamp' and memItem['children'][0]['attributes']['value'] == 'block':
            memStartPos,memEndPos = srcToPos(memItem['src'])
            location.append((memStartPos,memEndPos))
    return location

# 2.2 根据合约名称和函数名称，判断block.timestamp所在的函数的位置
def getLocation(ast_json, contractName, functionName):
    result = []
    contractAST = findASTNode(ast_json, 'name', 'ContractDefinition')
    bt_locations = getBlockTimeStamp(ast_json)
    for contractItem in contractAST:
        if contractItem['attributes']['name'] != contractName:
            continue
        functionAST = findASTNode(contractItem, 'name', 'FunctionDefinition')
        for functionItem in functionAST:
            if functionItem['attributes']['name'] != functionName:
                continue

            funcStartPos, funcEndPos = srcToPos(functionItem['src'])
            for start, end in bt_locations:
                if start >= funcStartPos and end <= funcEndPos:
                    result.append((funcStartPos, funcEndPos, FLAG_FUNC))

        modifierAST = findASTNode(contractItem, 'name', 'ModifierDefinition')
        for modifierItem in modifierAST:
            if modifierItem['attributes']['name'] != functionName:
                continue
            funcStartPos, funcEndPos = srcToPos(modifierItem['src'])
            for start, end in bt_locations:
                if start >= funcStartPos and end <= funcEndPos:
                    result.append((funcStartPos, funcEndPos, FLAG_FUNC))
    return result

##################################################################################
#
#   # 1
#
##################################################################################
def getConditionBlockTime(ast_json, contractName, functionName):
    '''
    Return tuples of either:
    - if statement location: (start byte, end byte, FLAG_IF)
    - variable declaration location: (start byte, end byte, FLAG_VARDEC)
    - assignment location: (start byte, end byte, FLAG_ASSIGN)
    '''
    contractAST = findASTNode(ast_json, 'name', 'ContractDefinition')
    locs = getBlockTimeStamp(ast_json)  #获取block.timestamp所在的位置
    ifLocation = []
    for contractItem in contractAST:
        if contractItem['attributes']['name'] != contractName:
            continue
        functionAST = findASTNode(contractItem, 'name', 'FunctionDefinition')
        for functionItem in functionAST:
            if functionItem['attributes']['name'] != functionName:
                continue;

            # 1.如果出现在条件判断语句当中
            ifAST = findASTNode(functionItem, 'name', 'IfStatement')
            for ifItem in ifAST:
                ifStartPos,ifEndPos = srcToPos(ifItem['src'])
                if ifItem["children"][0]["name"] == 'BinaryOperation':
                    binaryStartPos,binaryEndPos = srcToPos(ifItem['children'][0]['src'])
                    for loc in locs:
                        if loc[0] >= binaryStartPos and loc[1] <= binaryEndPos:
                            ifLocation.append((ifStartPos, ifEndPos, FLAG_IF))
                        else:
                            continue
                else:
                    continue

            # 2.出现在变量声明的语句当中
            variableDecAST = findASTNode(functionItem, 'name', 'VariableDeclarationStatement')
            for variableDecItem in variableDecAST:
                variableDecItemStartPos,variableDecItemEndPos = srcToPos(variableDecItem['src'])
                for loc in locs:
                    if loc[0] >= variableDecItemStartPos and loc[1] <= variableDecItemEndPos:
                        ifLocation.append((variableDecItemStartPos, variableDecItemEndPos, FLAG_VARDEC))
                    else:
                        continue

            # 3.出现在赋值变量语句当中
            assignAST = findASTNode(functionItem, 'name', 'Assignment')
            for assignItem in assignAST:
                assignStartPos,assignEndPos = srcToPos(assignItem['src'])
                for loc in locs:
                    if loc[0] >= assignStartPos and loc[1] <= assignEndPos:
                        ifLocation.append((assignStartPos, assignEndPos, FLAG_ASSIGN))
                    else:
                        continue

            # 4.出现在require语句当中
            functionCallAST = findASTNode(functionItem, 'name', 'FunctionCall')
            for functionItem in functionCallAST:
                functionCallStartPos,functionCallEndPos = srcToPos(functionItem['src'])
                for loc in locs:
                    if loc[0] >= functionCallStartPos and loc[1] <= functionCallEndPos:
                        ifLocation.append((functionCallStartPos, functionCallEndPos, FLAG_REQUIRE))
                    else:
                        continue


    return ifLocation



def getFuncBlockTime(ast_json, chains):
    result = []
    for chain in chains:
        for onepathItem in chain:
            contractName = onepathItem.split('.')[0]
            funcName = onepathItem.split('.')[1]
            res = getLocation(ast_json, contractName, funcName)
            if res:
                result.append(res)
    loList = sorted([n for a in result for n in a])
    return loList


##################################################################################
#
#   # 2
#
##################################################################################
def getConditionVar(ast_json, contractName, functionName):
    varList = []
    conditionBT = getConditionBlockTime(ast_json,contractName,functionName)
    contractAST = findASTNode(ast_json, 'name', 'ContractDefinition')
    for contractItem in contractAST:
        if contractItem['attributes']['name'] != contractName:
            continue
        functionAST = findASTNode(contractItem, 'name', 'FunctionDefinition')
        for functionItem in functionAST:
            if functionItem['attributes']['name'] != functionName:
                continue

            identifierAST = findASTNode(functionItem, 'name', 'Identifier')
            varDecAST = findASTNode(functionItem, 'name', 'VariableDeclaration')
            assignAST = findASTNode(functionItem, 'name', 'Assignment')

            # 3.2.2.1 存在于条件判断语句当中
            cond_filtered_if = [item for item in conditionBT if item[2] == FLAG_IF]
            for identifierItem in identifierAST:
                idenStartPos,idenEndPos = srcToPos(identifierItem['src'])
                for conditionBTItem in cond_filtered_if:
                    if idenStartPos >= conditionBTItem[0] and idenEndPos <= conditionBTItem[1]:
                        if identifierItem['attributes']['referencedDeclaration'] is not None and identifierItem['attributes']['referencedDeclaration'] > 0:
                            varList.append(identifierItem['attributes']['referencedDeclaration'])
                        else:
                            continue
                    else:
                        continue

            # 3.2.2.2 存在于正常的变量声明当中
            cond_filtered_vardec = [item for item in conditionBT if item[2] == FLAG_VARDEC]
            for varDecItem in varDecAST:
                varDecStartPos,varDecEndPos = srcToPos(varDecItem['src'])
                for conditionBTItem in cond_filtered_vardec:
                    if varDecStartPos >= conditionBTItem[0] and varDecEndPos <= conditionBTItem[1]:
                        varList.append(varDecItem['id'])

            # 3.2.2.3 存在于正常的变量赋值当中
            cond_filtered_assign = [item for item in conditionBT if item[2] == FLAG_ASSIGN]
            for assignItem in assignAST:
                assignStartPos,assignEndPos = srcToPos(assignItem['src'])
                for conditionBTItem in cond_filtered_assign:
                    if assignStartPos >= conditionBTItem[0] and assignEndPos <= conditionBTItem[1]:
                        if assignItem['attributes']['operator'] == '=':
                            if assignItem['children'][0]['attributes']['referencedDeclaration']:
                                varList.append(assignItem['children'][0]['attributes']['referencedDeclaration'])
                            else:
                                varList.append(assignItem['children'][0]['id'])
                        else:
                            continue
                    else:
                        continue
    return varList

def getConditionVarRelatedSC(ast_json, contractName, functionName):
    result = []
    varList = getConditionVar(ast_json,contractName,functionName)
    contractAST = findASTNode(ast_json, 'name', 'ContractDefinition')
    # 接下来所有id为这些东西的
    for contractItem in contractAST:
        if contractItem['attributes']['name'] != contractName:
            continue
        functionAST = findASTNode(contractItem, 'name', 'FunctionDefinition')
        for functionItem in functionAST:
            if functionItem['attributes']['name'] != functionName:
                continue

            identifierAST = findASTNode(functionItem, 'name', 'Identifier')
            memberAccessAST = findASTNode(functionItem, 'name', 'MemberAccess')
            varDecAST = findASTNode(functionItem, 'name', 'VariableDeclaration')
            for identifierItem in identifierAST:
                if identifierItem['id'] in varList or identifierItem['attributes']['referencedDeclaration'] in varList:
                    # idenStartPos,idenEndPos = srcToPos(identifierItem['src'])
                    idenPos = srcToFirstPos(identifierItem['src'])
                    result.append((idenPos,))
            for memberAccessItem in memberAccessAST:
                if memberAccessItem['id'] in varList or memberAccessItem['attributes']['referencedDeclaration'] in varList:
                    memberAccessPos = srcToFirstPos(memberAccessItem['src'])
                    result.append((memberAccessPos,))
            for varDecASTItem in varDecAST:
                if varDecASTItem['id'] in varList:
                    # idenStartPos,idenEndPos = srcToPos(identifierItem['src'])
                    idenPos = srcToFirstPos(varDecASTItem['src'])
                    result.append((idenPos,))
    return result

##################################################################################
#
#   # 3
#
##################################################################################
# 2.5 判断当前的这个合约的这个函数是不是modifier,如果是的话，我可以直接把整个的函数路径全部提取出来了
def isModifier(ast_json, contractName, functionName):
    result = []
    contractAST = findASTNode(ast_json, 'name', 'ContractDefinition')
    for contractItem in contractAST:
        if contractItem['attributes']['name'] != contractName:
            continue
        functionAST = findASTNode(contractItem, 'name', 'ModifierDefinition')
        for functionItem in functionAST:
            if functionItem['attributes']['name'] == functionName:
                funcStartPos,funcEndPos = srcToPos(functionItem['src'])
                result.append((funcStartPos,funcEndPos,FLAG_MODIFIER))
    return result


def getConditionRelatedSC(ast_json, chains, dot_files, root_dir):
    '''
    Used to slice time stamp dependency attack code.
    '''
    chains = getMaliciousChains(ast_json, chains, dot_files, root_dir, getLocation)
    result = []
    for chain in chains:
        for onepathItem in chain:
            contractName = onepathItem.split('.')[0]
            functionName = onepathItem.split('.')[1]

            # 1
            condition = getConditionBlockTime(ast_json, contractName, functionName)
            if len(condition) > 0:
                result.extend(condition)
                # print(f"cd: ", result)

            # 2
            sc = getConditionVarRelatedSC(ast_json, contractName, functionName)
            if len(sc) > 0:
                result.extend(sc)
                # print(f"sc: ", result)

            # 3
            modifier = isModifier(ast_json, contractName, functionName)
            if modifier:
                result.extend(modifier)
                # print(f"mod:", result)

    # print(result)
    fun = getFuncBlockTime(ast_json, chains)
    result.extend(fun)

    # item_ = [a for item in result for a in item]
    item_ = result
    # print(f"items {item_}")
    result_ = []
    for item in item_:
        if len(item) == 1:
            result_.append(item[0])
        elif item[2] == FLAG_IF:
            for i in range(item[0], item[1] + 1):
                result_.append(i)
        elif item[2] == FLAG_MODIFIER:
            for i in range(item[0],item[1]+1):
                result_.append(i)
        elif item[2] == FLAG_FUNC:
            result_.append(item[0])
            result_.append(item[1])

    print(result_)
    return result_

