import directory_transform from "./directory_transform.ts";

const input_dir = process.env.CORE_SRC_DIR!;
const output_dir = process.env.CORE_PREPROC_OUT_DIR!;

console.log("RUNNING C PREPROCESSOR.");
console.log(`  INPUT=${input_dir}`);
console.log(`  OUTPUT=${output_dir}\n`);
directory_transform(input_dir, output_dir, preprocess);
console.log("\nPREPROCESSING DONE.");

function multiline_macros(file_content: string): string {
    return file_content.replace(/(?<=(#define.*))\[\[\[[\s\S]*?\]\]\]/gm, (match: string) => match
        // remove [[[ and ]]] before and after the macro
        .slice(3, match.length - 3)

        // escape comments
        .replace(/\/\/+.*/gm, (match: string) => `/* ${match} */`)

        // replace newlines with \
        .replace(/\r\n|\r|\n/gm, " \\\n")
    );
}

function macro_invoke_generation(file_content: string): string {
    let processed = file_content;
    const assignments: {[a: string]: string} = {
        "=": "",
        "+=": "_acc",
    };

    processed = file_content.replace(/@GENERATE\s+\S+\s*\[\[\[[\s\S]*?\]\]\]/gm, (match: string) => {
        console.log(match);

        const macro_name = match
            .split(/\[\[\[/)[0]
            .slice(9).trim()
            .slice(1,-1).trim();
    
        const defined_operations = match
            .replace(/(@GENERATE\s+\S+\s*\[\[\[)|(\]\]\])/gm, "")
            .split(/\r\n|\r|\n/gm)
            .filter(op => op.trim().length !== 0)
            .map((op: string) => {
                return op
                    .split(/:(.+)/)
                    .slice(0, 2)
                    .map(part => part.trim());
            });

        const generated_code = defined_operations.map(([name, result]) => {
            return Object.keys(assignments)
                .map((assignment_type) => {
                    const postfix = assignments[assignment_type];
                    return `${macro_name}(${name}${postfix}, ${assignment_type}, ${result})`;
                })
                .join("\n");
        });

        return generated_code.join("\n\n");
    });

    return processed;
}

// file-by-file processing
function preprocess(file_content: string, file_name: string, path: string) {
    if (!/\.(c|h|cpp|hpp)$/.test(file_name))
        console.warn(`PREPROCESSOR [SKIP FILE]: Found a non-C file: ${path}.`);

    let processed = file_content;

    processed = multiline_macros(processed);
    processed = macro_invoke_generation(processed);

    return { content: processed, name: file_name };
}
