import sys
from antlr4 import FileStream, CommonTokenStream, ParseTreeWalker
from SolidityLexer import SolidityLexer
from SolidityParser import SolidityParser
from SolidityListener import SolidityListener

def parse(argv):
    input_stream = FileStream(argv[1])
    lexer = SolidityLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = SolidityParser(stream)
    tree = parser.sourceUnit()
    print(tree.toStringTree(recog=parser))

if __name__ == '__main__':
    parse(sys.argv)
