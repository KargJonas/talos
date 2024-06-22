import {rimraf} from "rimraf";
import fs from "fs";

/**
 * @description Recursively step through a folder, and it's sub-folders and apply a converter-function to the text-content of each one of the files.
 * @param {String} inDir Input folder.
 * @param {String} outDir Output folder.
 * @param {Function} convertFunc Function used to convert the content of each file. (convertFunc: (fileContent, fileName) => ({newFileContent, newFileName}))
 */
export default function directory_transform(
    inDir: string,
    outDir: string,
    convertFunc: (content: string, name: string, path: string) => {content: string, name: string}
) {
    // Throwing an exception if the input folder does not exist.
    if (!fs.existsSync(inDir) || !fs.lstatSync(inDir).isDirectory()) {
        throw new Error("The input folder is invalid.");
    }

    if (typeof convertFunc !== "function") {
        throw new Error("Invalid convert-function.");
    }

    // Creating the output folder
    if (fs.existsSync(outDir)) {
        rimraf.sync(outDir);
    }

    fs.mkdirSync(outDir);

    // Reading all of the items in the input directory.
    const items = fs.readdirSync(inDir);

    // Mapping through each of the elements
    for (const item of items) {
        const inPath = `${ inDir }/${ item }`;
        const stats = fs.lstatSync(inPath);

        // Applying the converter-function to files
        if (stats.isFile()) {
            const fileContent = fs.readFileSync(inPath).toString();
            const converted = convertFunc(fileContent, item, inPath);

            // Skipping files that don't return a proper value.
            if (
                typeof converted !== "object" ||
                !(typeof converted.name === "string" || typeof converted.name === "undefined")
            ) {
                console.warn(`Skipping file "${ item }" because of invalid return value.`);
                continue;
            }

            const convertedContent = converted.content;
            const convertedName = converted.name || item;

            const outPath = `${ outDir }/${ convertedName || item }`;
            fs.writeFileSync(outPath, convertedContent);
        }

        // Calling transform() on folders. (recursive)
        else if (stats.isDirectory()) {
            const outPath = `${ outDir }/${ item }`;
            fs.mkdirSync(outPath);
            transform(inPath, outPath, convertFunc);
        }
    }
}
