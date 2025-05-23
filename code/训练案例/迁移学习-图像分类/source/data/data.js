const loadImg = src=> {
	return new Promise((resolve, reject)=> {
		const img = new Image();
		img.crossOrigin = 'anonymous';
		img.src = src;
		img.width = 224;
		img.height = 224;
		img.onload = ()=> {
			resolve(img)
		}
		img.onerror = ()=> {
			reject(new Error('图片加载失败'))
		}
	})
}
export const getInputs = async ()=> {
	const loadImgs = [];
	const labels = [];
	for(let i=1; i<=20; i++) {
		['android', 'apple', 'windows'].forEach(label=> {
			const imgP = loadImg('./data/brand/train/'+label+'/'+i+'.jpeg');
			loadImgs.push(imgP);
			labels.push([
				label === 'android' ? 1 : 0,
				label === 'apple' ? 1 : 0,
				label === 'windows' ? 1 : 0
			]);
		});
	}
	const inputs = await Promise.all(loadImgs);
	return {
		inputs,
		labels
	}
}