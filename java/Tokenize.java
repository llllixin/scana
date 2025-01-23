import org.antlr.v4.runtime.ANTLRInputStream;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeWalker;

import java.io.*;

public class Tokenize {
    public static String result;
    public String getParseString(String sourcePath) throws IOException {
        result=" ";

        InputStream is = new FileInputStream(sourcePath);
        ANTLRInputStream input = new ANTLRInputStream(is);

        SolidityLexer lexer = new SolidityLexer(input);
        CommonTokenStream tokens = new CommonTokenStream(lexer);

        tokens.fill();

        SolidityParser parser = new SolidityParser(tokens);


        ParseTree tree = parser.sourceUnit();


        /* Extract Function Tokens */
        ParseTreeWalker walker = new ParseTreeWalker();

		// ASTSerialize listener = new ASTSerialize();
		// SolidityToSeq listener = new SolidityToSeq();
		test listener = new test();
		// ASTSerialize listener = new ASTSerialize();
		walker.walk(listener,tree);
        return result;
    }



    public static void main(String[] args) throws IOException {
		Tokenize tokenize = new Tokenize();

        try {
            if (args.length > 0) {
                // Read from file
                File file = new File(args[0]);
                if (!file.exists()) {
                    System.err.println("Error: File not found -> " + args[0]);
                    System.exit(1);
                }
            }

            // Process the input
			String parsedResult = tokenize.getParseString(args[0]);
            System.out.println(parsedResult);

        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
		return;
//         String filepath = "C:\\Users\\13548\\Desktop\\new\\";
// //        String filepath = "src/main/java/test/";
// //        String targetPath = "E:\\maven\\kuangshenstudy\\SmartContracts\\SolidityANTLR\\src\\main\\ASTSerialize\\";
//         String targetPath = "D:\\AST\\src\\main\\java\\ASTSerialize\\";
//         Tokenize tokenize = new Tokenize();
//
//
//         File file = new File(filepath);
//         File[] tempList = file.listFiles();
//         System.out.println(tempList.length);
//         try {
//             for (int i = 0; i < tempList.length; i++) {
//                 System.out.println(tempList[i].getName());
//                 String s = tokenize.getParseString(filepath + tempList[i].getName());
//                 System.out.println(Integer.toString(i) + s);
//
// //                File f = new File(targetPath + tempList[i].getName());
// //                if (!f.exists()){
// //                    f.createNewFile();
// //                }
//                 FileWriter fileWriter = new FileWriter(targetPath + tempList[i].getName(),true);
//                 fileWriter.write(s);
//                 fileWriter.close();
//             }
//         } catch (IOException e){
//             e.printStackTrace();
//         }
    }

}
