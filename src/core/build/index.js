var Module=typeof Module!="undefined"?Module:{};var moduleOverrides=Object.assign({},Module);var arguments_=[];var thisProgram="./this.program";var quit_=(status,toThrow)=>{throw toThrow};var ENVIRONMENT_IS_WEB=typeof window=="object";var ENVIRONMENT_IS_WORKER=typeof importScripts=="function";var ENVIRONMENT_IS_NODE=typeof process=="object"&&typeof process.versions=="object"&&typeof process.versions.node=="string";var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var read_,readAsync,readBinary,setWindowTitle;function logExceptionOnExit(e){if(e instanceof ExitStatus)return;let toLog=e;err("exiting due to exception: "+toLog)}var fs;var nodePath;var requireNodeFS;if(ENVIRONMENT_IS_NODE){if(ENVIRONMENT_IS_WORKER){scriptDirectory=require("path").dirname(scriptDirectory)+"/"}else{scriptDirectory=__dirname+"/"}requireNodeFS=(()=>{if(!nodePath){fs=require("fs");nodePath=require("path")}});read_=function shell_read(filename,binary){requireNodeFS();filename=nodePath["normalize"](filename);return fs.readFileSync(filename,binary?undefined:"utf8")};readBinary=(filename=>{var ret=read_(filename,true);if(!ret.buffer){ret=new Uint8Array(ret)}return ret});readAsync=((filename,onload,onerror)=>{requireNodeFS();filename=nodePath["normalize"](filename);fs.readFile(filename,function(err,data){if(err)onerror(err);else onload(data.buffer)})});if(process["argv"].length>1){thisProgram=process["argv"][1].replace(/\\/g,"/")}arguments_=process["argv"].slice(2);if(typeof module!="undefined"){module["exports"]=Module}process["on"]("uncaughtException",function(ex){if(!(ex instanceof ExitStatus)){throw ex}});process["on"]("unhandledRejection",function(reason){throw reason});quit_=((status,toThrow)=>{if(keepRuntimeAlive()){process["exitCode"]=status;throw toThrow}logExceptionOnExit(toThrow);process["exit"](status)});Module["inspect"]=function(){return"[Emscripten Module object]"}}else if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){if(ENVIRONMENT_IS_WORKER){scriptDirectory=self.location.href}else if(typeof document!="undefined"&&document.currentScript){scriptDirectory=document.currentScript.src}if(scriptDirectory.indexOf("blob:")!==0){scriptDirectory=scriptDirectory.substr(0,scriptDirectory.replace(/[?#].*/,"").lastIndexOf("/")+1)}else{scriptDirectory=""}{read_=(url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.send(null);return xhr.responseText});if(ENVIRONMENT_IS_WORKER){readBinary=(url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)})}readAsync=((url,onload,onerror)=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,true);xhr.responseType="arraybuffer";xhr.onload=(()=>{if(xhr.status==200||xhr.status==0&&xhr.response){onload(xhr.response);return}onerror()});xhr.onerror=onerror;xhr.send(null)})}setWindowTitle=(title=>document.title=title)}else{}var out=Module["print"]||console.log.bind(console);var err=Module["printErr"]||console.warn.bind(console);Object.assign(Module,moduleOverrides);moduleOverrides=null;if(Module["arguments"])arguments_=Module["arguments"];if(Module["thisProgram"])thisProgram=Module["thisProgram"];if(Module["quit"])quit_=Module["quit"];var wasmBinary;if(Module["wasmBinary"])wasmBinary=Module["wasmBinary"];var noExitRuntime=Module["noExitRuntime"]||true;if(typeof WebAssembly!="object"){abort("no native wasm support detected")}function setValue(ptr,value,type="i8",noSafe){if(type.charAt(type.length-1)==="*")type="i32";switch(type){case"i1":HEAP8[ptr>>0]=value;break;case"i8":HEAP8[ptr>>0]=value;break;case"i16":HEAP16[ptr>>1]=value;break;case"i32":HEAP32[ptr>>2]=value;break;case"i64":tempI64=[value>>>0,(tempDouble=value,+Math.abs(tempDouble)>=1?tempDouble>0?(Math.min(+Math.floor(tempDouble/4294967296),4294967295)|0)>>>0:~~+Math.ceil((tempDouble-+(~~tempDouble>>>0))/4294967296)>>>0:0)],HEAP32[ptr>>2]=tempI64[0],HEAP32[ptr+4>>2]=tempI64[1];break;case"float":HEAPF32[ptr>>2]=value;break;case"double":HEAPF64[ptr>>3]=value;break;default:abort("invalid type for setValue: "+type)}}function getValue(ptr,type="i8",noSafe){if(type.charAt(type.length-1)==="*")type="i32";switch(type){case"i1":return HEAP8[ptr>>0];case"i8":return HEAP8[ptr>>0];case"i16":return HEAP16[ptr>>1];case"i32":return HEAP32[ptr>>2];case"i64":return HEAP32[ptr>>2];case"float":return HEAPF32[ptr>>2];case"double":return Number(HEAPF64[ptr>>3]);default:abort("invalid type for getValue: "+type)}return null}var wasmMemory;var ABORT=false;var EXITSTATUS;function getCFunc(ident){var func=Module["_"+ident];return func}function ccall(ident,returnType,argTypes,args,opts){var toC={"string":function(str){var ret=0;if(str!==null&&str!==undefined&&str!==0){var len=(str.length<<2)+1;ret=stackAlloc(len);stringToUTF8(str,ret,len)}return ret},"array":function(arr){var ret=stackAlloc(arr.length);writeArrayToMemory(arr,ret);return ret}};function convertReturnValue(ret){if(returnType==="string")return UTF8ToString(ret);if(returnType==="boolean")return Boolean(ret);return ret}var func=getCFunc(ident);var cArgs=[];var stack=0;if(args){for(var i=0;i<args.length;i++){var converter=toC[argTypes[i]];if(converter){if(stack===0)stack=stackSave();cArgs[i]=converter(args[i])}else{cArgs[i]=args[i]}}}var ret=func.apply(null,cArgs);function onDone(ret){if(stack!==0)stackRestore(stack);return convertReturnValue(ret)}ret=onDone(ret);return ret}function cwrap(ident,returnType,argTypes,opts){argTypes=argTypes||[];var numericArgs=argTypes.every(function(type){return type==="number"});var numericRet=returnType!=="string";if(numericRet&&numericArgs&&!opts){return getCFunc(ident)}return function(){return ccall(ident,returnType,argTypes,arguments,opts)}}var UTF8Decoder=typeof TextDecoder!="undefined"?new TextDecoder("utf8"):undefined;function UTF8ArrayToString(heap,idx,maxBytesToRead){var endIdx=idx+maxBytesToRead;var endPtr=idx;while(heap[endPtr]&&!(endPtr>=endIdx))++endPtr;if(endPtr-idx>16&&heap.subarray&&UTF8Decoder){return UTF8Decoder.decode(heap.subarray(idx,endPtr))}else{var str="";while(idx<endPtr){var u0=heap[idx++];if(!(u0&128)){str+=String.fromCharCode(u0);continue}var u1=heap[idx++]&63;if((u0&224)==192){str+=String.fromCharCode((u0&31)<<6|u1);continue}var u2=heap[idx++]&63;if((u0&240)==224){u0=(u0&15)<<12|u1<<6|u2}else{u0=(u0&7)<<18|u1<<12|u2<<6|heap[idx++]&63}if(u0<65536){str+=String.fromCharCode(u0)}else{var ch=u0-65536;str+=String.fromCharCode(55296|ch>>10,56320|ch&1023)}}}return str}function UTF8ToString(ptr,maxBytesToRead){return ptr?UTF8ArrayToString(HEAPU8,ptr,maxBytesToRead):""}function stringToUTF8Array(str,heap,outIdx,maxBytesToWrite){if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.charCodeAt(i);if(u>=55296&&u<=57343){var u1=str.charCodeAt(++i);u=65536+((u&1023)<<10)|u1&1023}if(u<=127){if(outIdx>=endIdx)break;heap[outIdx++]=u}else if(u<=2047){if(outIdx+1>=endIdx)break;heap[outIdx++]=192|u>>6;heap[outIdx++]=128|u&63}else if(u<=65535){if(outIdx+2>=endIdx)break;heap[outIdx++]=224|u>>12;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63}else{if(outIdx+3>=endIdx)break;heap[outIdx++]=240|u>>18;heap[outIdx++]=128|u>>12&63;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63}}heap[outIdx]=0;return outIdx-startIdx}function stringToUTF8(str,outPtr,maxBytesToWrite){return stringToUTF8Array(str,HEAPU8,outPtr,maxBytesToWrite)}function writeArrayToMemory(array,buffer){HEAP8.set(array,buffer)}function alignUp(x,multiple){if(x%multiple>0){x+=multiple-x%multiple}return x}var buffer,HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64;function updateGlobalBufferAndViews(buf){buffer=buf;Module["HEAP8"]=HEAP8=new Int8Array(buf);Module["HEAP16"]=HEAP16=new Int16Array(buf);Module["HEAP32"]=HEAP32=new Int32Array(buf);Module["HEAPU8"]=HEAPU8=new Uint8Array(buf);Module["HEAPU16"]=HEAPU16=new Uint16Array(buf);Module["HEAPU32"]=HEAPU32=new Uint32Array(buf);Module["HEAPF32"]=HEAPF32=new Float32Array(buf);Module["HEAPF64"]=HEAPF64=new Float64Array(buf)}var INITIAL_MEMORY=Module["INITIAL_MEMORY"]||16777216;var wasmTable;var __ATPRERUN__=[];var __ATINIT__=[];var __ATPOSTRUN__=[];var runtimeInitialized=false;var runtimeKeepaliveCounter=0;function keepRuntimeAlive(){return noExitRuntime||runtimeKeepaliveCounter>0}function preRun(){if(Module["preRun"]){if(typeof Module["preRun"]=="function")Module["preRun"]=[Module["preRun"]];while(Module["preRun"].length){addOnPreRun(Module["preRun"].shift())}}callRuntimeCallbacks(__ATPRERUN__)}function initRuntime(){runtimeInitialized=true;callRuntimeCallbacks(__ATINIT__)}function postRun(){if(Module["postRun"]){if(typeof Module["postRun"]=="function")Module["postRun"]=[Module["postRun"]];while(Module["postRun"].length){addOnPostRun(Module["postRun"].shift())}}callRuntimeCallbacks(__ATPOSTRUN__)}function addOnPreRun(cb){__ATPRERUN__.unshift(cb)}function addOnInit(cb){__ATINIT__.unshift(cb)}function addOnPostRun(cb){__ATPOSTRUN__.unshift(cb)}var runDependencies=0;var runDependencyWatcher=null;var dependenciesFulfilled=null;function addRunDependency(id){runDependencies++;if(Module["monitorRunDependencies"]){Module["monitorRunDependencies"](runDependencies)}}function removeRunDependency(id){runDependencies--;if(Module["monitorRunDependencies"]){Module["monitorRunDependencies"](runDependencies)}if(runDependencies==0){if(runDependencyWatcher!==null){clearInterval(runDependencyWatcher);runDependencyWatcher=null}if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback()}}}Module["preloadedImages"]={};Module["preloadedAudios"]={};function abort(what){{if(Module["onAbort"]){Module["onAbort"](what)}}what="Aborted("+what+")";err(what);ABORT=true;EXITSTATUS=1;what+=". Build with -s ASSERTIONS=1 for more info.";var e=new WebAssembly.RuntimeError(what);throw e}var dataURIPrefix="data:application/octet-stream;base64,";function isDataURI(filename){return filename.startsWith(dataURIPrefix)}function isFileURI(filename){return filename.startsWith("file://")}var wasmBinaryFile;wasmBinaryFile="index.wasm";if(!isDataURI(wasmBinaryFile)){wasmBinaryFile=locateFile(wasmBinaryFile)}function getBinary(file){try{if(file==wasmBinaryFile&&wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(file)}else{throw"both async and sync fetching of the wasm failed"}}catch(err){abort(err)}}function getBinaryPromise(){if(!wasmBinary&&(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER)){if(typeof fetch=="function"&&!isFileURI(wasmBinaryFile)){return fetch(wasmBinaryFile,{credentials:"same-origin"}).then(function(response){if(!response["ok"]){throw"failed to load wasm binary file at '"+wasmBinaryFile+"'"}return response["arrayBuffer"]()}).catch(function(){return getBinary(wasmBinaryFile)})}else{if(readAsync){return new Promise(function(resolve,reject){readAsync(wasmBinaryFile,function(response){resolve(new Uint8Array(response))},reject)})}}}return Promise.resolve().then(function(){return getBinary(wasmBinaryFile)})}function createWasm(){var info={"a":asmLibraryArg};function receiveInstance(instance,module){var exports=instance.exports;Module["asm"]=exports;wasmMemory=Module["asm"]["c"];updateGlobalBufferAndViews(wasmMemory.buffer);wasmTable=Module["asm"]["ea"];addOnInit(Module["asm"]["d"]);removeRunDependency("wasm-instantiate")}addRunDependency("wasm-instantiate");function receiveInstantiationResult(result){receiveInstance(result["instance"])}function instantiateArrayBuffer(receiver){return getBinaryPromise().then(function(binary){return WebAssembly.instantiate(binary,info)}).then(function(instance){return instance}).then(receiver,function(reason){err("failed to asynchronously prepare wasm: "+reason);abort(reason)})}function instantiateAsync(){if(!wasmBinary&&typeof WebAssembly.instantiateStreaming=="function"&&!isDataURI(wasmBinaryFile)&&!isFileURI(wasmBinaryFile)&&typeof fetch=="function"){return fetch(wasmBinaryFile,{credentials:"same-origin"}).then(function(response){var result=WebAssembly.instantiateStreaming(response,info);return result.then(receiveInstantiationResult,function(reason){err("wasm streaming compile failed: "+reason);err("falling back to ArrayBuffer instantiation");return instantiateArrayBuffer(receiveInstantiationResult)})})}else{return instantiateArrayBuffer(receiveInstantiationResult)}}if(Module["instantiateWasm"]){try{var exports=Module["instantiateWasm"](info,receiveInstance);return exports}catch(e){err("Module.instantiateWasm callback failed with error: "+e);return false}}instantiateAsync();return{}}var tempDouble;var tempI64;function callRuntimeCallbacks(callbacks){while(callbacks.length>0){var callback=callbacks.shift();if(typeof callback=="function"){callback(Module);continue}var func=callback.func;if(typeof func=="number"){if(callback.arg===undefined){getWasmTableEntry(func)()}else{getWasmTableEntry(func)(callback.arg)}}else{func(callback.arg===undefined?null:callback.arg)}}}var wasmTableMirror=[];function getWasmTableEntry(funcPtr){var func=wasmTableMirror[funcPtr];if(!func){if(funcPtr>=wasmTableMirror.length)wasmTableMirror.length=funcPtr+1;wasmTableMirror[funcPtr]=func=wasmTable.get(funcPtr)}return func}function _emscripten_memcpy_big(dest,src,num){HEAPU8.copyWithin(dest,src,src+num)}function _emscripten_get_heap_max(){return 2147483648}function emscripten_realloc_buffer(size){try{wasmMemory.grow(size-buffer.byteLength+65535>>>16);updateGlobalBufferAndViews(wasmMemory.buffer);return 1}catch(e){}}function _emscripten_resize_heap(requestedSize){var oldSize=HEAPU8.length;requestedSize=requestedSize>>>0;var maxHeapSize=_emscripten_get_heap_max();if(requestedSize>maxHeapSize){return false}for(var cutDown=1;cutDown<=4;cutDown*=2){var overGrownHeapSize=oldSize*(1+.2/cutDown);overGrownHeapSize=Math.min(overGrownHeapSize,requestedSize+100663296);var newSize=Math.min(maxHeapSize,alignUp(Math.max(requestedSize,overGrownHeapSize),65536));var replacement=emscripten_realloc_buffer(newSize);if(replacement){return true}}return false}var asmLibraryArg={"b":_emscripten_memcpy_big,"a":_emscripten_resize_heap};var asm=createWasm();var ___wasm_call_ctors=Module["___wasm_call_ctors"]=function(){return(___wasm_call_ctors=Module["___wasm_call_ctors"]=Module["asm"]["d"]).apply(null,arguments)};var _alloc_farr=Module["_alloc_farr"]=function(){return(_alloc_farr=Module["_alloc_farr"]=Module["asm"]["e"]).apply(null,arguments)};var _alloc_starr=Module["_alloc_starr"]=function(){return(_alloc_starr=Module["_alloc_starr"]=Module["asm"]["f"]).apply(null,arguments)};var _copy_farr=Module["_copy_farr"]=function(){return(_copy_farr=Module["_copy_farr"]=Module["asm"]["g"]).apply(null,arguments)};var _copy_starr=Module["_copy_starr"]=function(){return(_copy_starr=Module["_copy_starr"]=Module["asm"]["h"]).apply(null,arguments)};var _free_farr=Module["_free_farr"]=function(){return(_free_farr=Module["_free_farr"]=Module["asm"]["i"]).apply(null,arguments)};var _free_starr=Module["_free_starr"]=function(){return(_free_starr=Module["_free_starr"]=Module["asm"]["j"]).apply(null,arguments)};var _free_tensor=Module["_free_tensor"]=function(){return(_free_tensor=Module["_free_tensor"]=Module["asm"]["k"]).apply(null,arguments)};var _rand_seed=Module["_rand_seed"]=function(){return(_rand_seed=Module["_rand_seed"]=Module["asm"]["l"]).apply(null,arguments)};var _rand_f=Module["_rand_f"]=function(){return(_rand_f=Module["_rand_f"]=Module["asm"]["m"]).apply(null,arguments)};var _rand_i=Module["_rand_i"]=function(){return(_rand_i=Module["_rand_i"]=Module["asm"]["n"]).apply(null,arguments)};var _fill=Module["_fill"]=function(){return(_fill=Module["_fill"]=Module["asm"]["o"]).apply(null,arguments)};var _negate_tns=Module["_negate_tns"]=function(){return(_negate_tns=Module["_negate_tns"]=Module["asm"]["p"]).apply(null,arguments)};var _sin_tns=Module["_sin_tns"]=function(){return(_sin_tns=Module["_sin_tns"]=Module["asm"]["q"]).apply(null,arguments)};var _cos_tns=Module["_cos_tns"]=function(){return(_cos_tns=Module["_cos_tns"]=Module["asm"]["r"]).apply(null,arguments)};var _tan_tns=Module["_tan_tns"]=function(){return(_tan_tns=Module["_tan_tns"]=Module["asm"]["s"]).apply(null,arguments)};var _asin_tns=Module["_asin_tns"]=function(){return(_asin_tns=Module["_asin_tns"]=Module["asm"]["t"]).apply(null,arguments)};var _acos_tns=Module["_acos_tns"]=function(){return(_acos_tns=Module["_acos_tns"]=Module["asm"]["u"]).apply(null,arguments)};var _atan_tns=Module["_atan_tns"]=function(){return(_atan_tns=Module["_atan_tns"]=Module["asm"]["v"]).apply(null,arguments)};var _sinh_tns=Module["_sinh_tns"]=function(){return(_sinh_tns=Module["_sinh_tns"]=Module["asm"]["w"]).apply(null,arguments)};var _cosh_tns=Module["_cosh_tns"]=function(){return(_cosh_tns=Module["_cosh_tns"]=Module["asm"]["x"]).apply(null,arguments)};var _tanh_tns=Module["_tanh_tns"]=function(){return(_tanh_tns=Module["_tanh_tns"]=Module["asm"]["y"]).apply(null,arguments)};var _exp_tns=Module["_exp_tns"]=function(){return(_exp_tns=Module["_exp_tns"]=Module["asm"]["z"]).apply(null,arguments)};var _log_tns=Module["_log_tns"]=function(){return(_log_tns=Module["_log_tns"]=Module["asm"]["A"]).apply(null,arguments)};var _log10_tns=Module["_log10_tns"]=function(){return(_log10_tns=Module["_log10_tns"]=Module["asm"]["B"]).apply(null,arguments)};var _log2_tns=Module["_log2_tns"]=function(){return(_log2_tns=Module["_log2_tns"]=Module["asm"]["C"]).apply(null,arguments)};var _invsqrt_tns=Module["_invsqrt_tns"]=function(){return(_invsqrt_tns=Module["_invsqrt_tns"]=Module["asm"]["D"]).apply(null,arguments)};var _sqrt_tns=Module["_sqrt_tns"]=function(){return(_sqrt_tns=Module["_sqrt_tns"]=Module["asm"]["E"]).apply(null,arguments)};var _ceil_tns=Module["_ceil_tns"]=function(){return(_ceil_tns=Module["_ceil_tns"]=Module["asm"]["F"]).apply(null,arguments)};var _floor_tns=Module["_floor_tns"]=function(){return(_floor_tns=Module["_floor_tns"]=Module["asm"]["G"]).apply(null,arguments)};var _abs_tns=Module["_abs_tns"]=function(){return(_abs_tns=Module["_abs_tns"]=Module["asm"]["H"]).apply(null,arguments)};var _reciprocal_tns=Module["_reciprocal_tns"]=function(){return(_reciprocal_tns=Module["_reciprocal_tns"]=Module["asm"]["I"]).apply(null,arguments)};var _identity_tns=Module["_identity_tns"]=function(){return(_identity_tns=Module["_identity_tns"]=Module["asm"]["J"]).apply(null,arguments)};var _relu_tns=Module["_relu_tns"]=function(){return(_relu_tns=Module["_relu_tns"]=Module["asm"]["K"]).apply(null,arguments)};var _binstep_tns=Module["_binstep_tns"]=function(){return(_binstep_tns=Module["_binstep_tns"]=Module["asm"]["L"]).apply(null,arguments)};var _logistic_tns=Module["_logistic_tns"]=function(){return(_logistic_tns=Module["_logistic_tns"]=Module["asm"]["M"]).apply(null,arguments)};var _sigmoid_tns=Module["_sigmoid_tns"]=function(){return(_sigmoid_tns=Module["_sigmoid_tns"]=Module["asm"]["N"]).apply(null,arguments)};var _add_scl=Module["_add_scl"]=function(){return(_add_scl=Module["_add_scl"]=Module["asm"]["O"]).apply(null,arguments)};var _sub_scl=Module["_sub_scl"]=function(){return(_sub_scl=Module["_sub_scl"]=Module["asm"]["P"]).apply(null,arguments)};var _mul_scl=Module["_mul_scl"]=function(){return(_mul_scl=Module["_mul_scl"]=Module["asm"]["Q"]).apply(null,arguments)};var _div_scl=Module["_div_scl"]=function(){return(_div_scl=Module["_div_scl"]=Module["asm"]["R"]).apply(null,arguments)};var _pow_scl=Module["_pow_scl"]=function(){return(_pow_scl=Module["_pow_scl"]=Module["asm"]["S"]).apply(null,arguments)};var _add_prw=Module["_add_prw"]=function(){return(_add_prw=Module["_add_prw"]=Module["asm"]["T"]).apply(null,arguments)};var _sub_prw=Module["_sub_prw"]=function(){return(_sub_prw=Module["_sub_prw"]=Module["asm"]["U"]).apply(null,arguments)};var _mul_prw=Module["_mul_prw"]=function(){return(_mul_prw=Module["_mul_prw"]=Module["asm"]["V"]).apply(null,arguments)};var _div_prw=Module["_div_prw"]=function(){return(_div_prw=Module["_div_prw"]=Module["asm"]["W"]).apply(null,arguments)};var _add_prw_brc=Module["_add_prw_brc"]=function(){return(_add_prw_brc=Module["_add_prw_brc"]=Module["asm"]["X"]).apply(null,arguments)};var _sub_prw_brc=Module["_sub_prw_brc"]=function(){return(_sub_prw_brc=Module["_sub_prw_brc"]=Module["asm"]["Y"]).apply(null,arguments)};var _mul_prw_brc=Module["_mul_prw_brc"]=function(){return(_mul_prw_brc=Module["_mul_prw_brc"]=Module["asm"]["Z"]).apply(null,arguments)};var _div_prw_brc=Module["_div_prw_brc"]=function(){return(_div_prw_brc=Module["_div_prw_brc"]=Module["asm"]["_"]).apply(null,arguments)};var _mul_tns=Module["_mul_tns"]=function(){return(_mul_tns=Module["_mul_tns"]=Module["asm"]["$"]).apply(null,arguments)};var _dot_tns=Module["_dot_tns"]=function(){return(_dot_tns=Module["_dot_tns"]=Module["asm"]["aa"]).apply(null,arguments)};var _create_tensor=Module["_create_tensor"]=function(){return(_create_tensor=Module["_create_tensor"]=Module["asm"]["ba"]).apply(null,arguments)};var _copy_tensor_metadata=Module["_copy_tensor_metadata"]=function(){return(_copy_tensor_metadata=Module["_copy_tensor_metadata"]=Module["asm"]["ca"]).apply(null,arguments)};var _copy_tensor=Module["_copy_tensor"]=function(){return(_copy_tensor=Module["_copy_tensor"]=Module["asm"]["da"]).apply(null,arguments)};var stackSave=Module["stackSave"]=function(){return(stackSave=Module["stackSave"]=Module["asm"]["fa"]).apply(null,arguments)};var stackRestore=Module["stackRestore"]=function(){return(stackRestore=Module["stackRestore"]=Module["asm"]["ga"]).apply(null,arguments)};var stackAlloc=Module["stackAlloc"]=function(){return(stackAlloc=Module["stackAlloc"]=Module["asm"]["ha"]).apply(null,arguments)};Module["ccall"]=ccall;Module["cwrap"]=cwrap;Module["setValue"]=setValue;Module["getValue"]=getValue;var calledRun;function ExitStatus(status){this.name="ExitStatus";this.message="Program terminated with exit("+status+")";this.status=status}dependenciesFulfilled=function runCaller(){if(!calledRun)run();if(!calledRun)dependenciesFulfilled=runCaller};function run(args){args=args||arguments_;if(runDependencies>0){return}preRun();if(runDependencies>0){return}function doRun(){if(calledRun)return;calledRun=true;Module["calledRun"]=true;if(ABORT)return;initRuntime();if(Module["onRuntimeInitialized"])Module["onRuntimeInitialized"]();postRun()}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(function(){setTimeout(function(){Module["setStatus"]("")},1);doRun()},1)}else{doRun()}}Module["run"]=run;if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].pop()()}}run();
