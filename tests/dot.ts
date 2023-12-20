import { core_ready } from '../src//util';
import tensor from '../src/tensor';

const print = (t) => console.log(t?.toString() + "\n---");

core_ready.then(() => {
    const d1 = [ -1, -0.3099989891052246, 0.5054184198379517, 0.5914905071258545, 0.554784893989563, -0.6215711832046509, -0.7422678470611572, 0.20738208293914795, -0.37373805046081543, 0.34012699127197266, -0.15614813566207886, 0.06877803802490234, 0.40990686416625977, -0.12000596523284912, 0.319480299949646, 0.9805681705474854, 0.08500480651855469, -0.09237104654312134, -0.8974905610084534, -0.07802224159240723, -0.4924488067626953, -0.12771075963974, 0.3332597017288208, -0.3075905442237854, 0.5569461584091187, 0.3619152307510376, -0.9680061936378479, -0.1414889097213745, -0.9010155200958252, 0.8770675659179688, -0.5507410764694214, -0.5678074359893799, 0.6062679290771484, -0.07216233015060425, 0.6642856597900391, 0.3953653573989868, -0.5615310668945312, -0.3478749394416809, 0.928863525390625, -0.05085664987564087, 0.3073357343673706, 0.5964829921722412, -0.21692383289337158, -0.6619696617126465, -0.1947472095489502, -0.057803332805633545, 0.06647336483001709, 0.8570867776870728, 0.9772183895111084, 0.8896461725234985, 0.5452755689620972, -0.13562828302383423, 0.45777273178100586, -0.19978058338165283, -0.12629443407058716, 0.821818470954895, -0.8142737746238708, -0.9137349724769592, 0.6381558179855347, -0.7661720514297485 ];
    const d2 = [ -0.3564334511756897, -0.22758489847183228, -0.02925652265548706, -0.26315420866012573, -0.9318785071372986, 0.25062990188598633, -0.751793384552002, -0.14405763149261475, 0.7440788745880127, -0.7868169546127319, -0.8113561272621155, -0.9898955225944519, -0.9688637256622314, 0.7444324493408203, -0.9608199000358582, -0.08647501468658447, -0.9258899688720703, -0.5853681564331055, -0.8150002360343933, -0.33617472648620605, 0.04223191738128662, 0.9862827062606812, -0.4453490972518921, -0.46856623888015747, 0.5298029184341431, 0.15364432334899902, -0.5243135690689087, 0.9345762729644775, 0.1539170742034912, 0.37051093578338623, 0.4680964946746826, -0.9635454416275024, -0.04870229959487915, -0.9078027606010437, -0.7243399620056152, -0.638577938079834, -0.16598069667816162, 0.060320138931274414, -0.9841264486312866, -0.13800519704818726 ];

    let t1 = tensor([3, 4, 5], d1);
    let t2 = tensor([2, 2, 5, 2], d2);

    print(t1.dot(t2))
    print(t1.dot(t2).shape)
});
