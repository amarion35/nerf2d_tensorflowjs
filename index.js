function resizeImage(imgToResize, height, width) {
  console.log("resize image")
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");

  canvas.width = width;
  canvas.height = height;

  context.drawImage(
    imgToResize,
    0,
    0,
    width,
    height
  );
  return canvas.toDataURL();
}

function getX(input){
  let X = []
  for (let i=0; i<input.shape[0]; i++) {
    for (let j=0; j<input.shape[1]; j++) {
      X.push([i,j])
    }
  }
  return tf.tensor(X)
}

function getY(input){
  let Y = input.slice([0,0,0],[-1,-1,3]);
  Y = Y.reshape([Y.shape[0]*Y.shape[1], 3])
  return Y
}

class PositionalEncoding {
  constructor(resolution) {
    this.resolution = resolution;
  }

  async fit(input) {
    this.min = input.min(0)
    this.max = input.max(0)
    this.range = 720
  }

  async transform(input) {
    console.log("transform")
      let x = input.sub(this.min)
      x = x.div(this.max.sub(this.min))
      let encoded_x = []
      for(let i=0; i<this.resolution; i++) {
        encoded_x.push(tf.cos(x.mul(Math.PI * (i+1))))
        encoded_x.push(tf.sin(x.mul(Math.PI * (i+1))))
      }
      return tf.concat(encoded_x, 1)
  }
}

class Normalize {
  async fit(input) {
    this.mean = input.mean(0)
    this.std = tf.tensor(math.std(input.arraySync(), 0))
  }

  async transform(input) {
    let x = input.sub(this.mean)
    x = x.div(this.std)
    return x
  }

  async invert(input) {
    let x = input.mul(this.std)
    x = x.add(this.mean)
    return x
  }
}

async function getModel(inputShape) {
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 1024, activation: 'relu', inputShape: inputShape[1]}));
  model.add(tf.layers.dense({units: 128, activation: 'relu'}));
  model.add(tf.layers.dense({units: 128, activation: 'relu'}));
  model.add(tf.layers.dense({units: 128, activation: 'relu'}));
  model.add(tf.layers.dense({units: 128, activation: 'relu'}));
  model.add(tf.layers.dense({units: 128, activation: 'relu'}));
  model.add(tf.layers.dense({units: 3}));
  model.compile({loss: 'meanSquaredError', optimizer: 'adam'});
  return model
}

async function generateImage(input){
  console.log(input.shape)
  let X = getX(input)
  let Y = getY(input)

  console.log(X.shape)
  console.log(Y.shape)

  // Positional encoding
  encoding = new PositionalEncoding(100)
  await encoding.fit(X)
  X = await encoding.transform(X)

  // Normalization
  norm = new Normalize()
  await norm.fit(Y)
  Y = await norm.transform(Y)

  // Init model
  const model = await getModel(X.shape)
  model.summary();

  // Inference
  batch_size = 256
  console.log("Inference")
  pred = await model.predict(X, {batchSize: batch_size})  
  await displayPred(pred, norm)

  // Train model
  console.log("Training")
  epochs = 100
  n_samples = X.shape[0]
  n_batches = Math.ceil(n_samples/batch_size)
  n_batches_percent = Math.floor(n_batches/100)
  progressBar = document.getElementById("epochProgress")
  epochField = document.getElementById("epoch")
  await model.fit(X, Y, {
    epochs: epochs, 
    batchSize: batch_size,
    shuffle: true, 
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        progressBar.value = 100*(batch/n_batches)
        if(batch%n_batches_percent==0){
          console.log(Math.round((batch/n_batches)*100)+"% - "+batch+"/"+n_batches);
        }
      },
      onEpochEnd: async (epoch, logs) => {
        console.log("Epoch "+ (epoch+1)+"/"+epochs+" : "+logs.loss);
        // Inference
        console.log("Inference")
        pred = await model.predict(X, {batchSize: batch_size})  
        await displayPred(pred, norm)
      },
      onEpochBegin: async (epoch, logs) => {
        epochField.innerHTML = "Training: Epoch "+(epoch+1)
      },
    }
  })
}

var outputHistory = []

async function displayPred(pred, norm) {
  // Invert normalization
  console.log("Invert")
  pred = await norm.invert(pred)

  // Make an image
  console.log("Casting")
  pred = pred.clipByValue(0,255)
  pred = tf.cast(pred, "int32")
  pred = pred.reshape([720,720,3])
  outputHistory.push(pred.clone())
  console.log(pred)

  let canvas = document.createElement('canvas');
  await tf.browser.toPixels(pred, canvas)
  document.getElementById('outputImage').src = canvas.toDataURL();
}

// Tiny TFJS train / predict example.
async function run() {
  // load image
  const img_path = "./bird2.jpg"
  const img = document.getElementById('inputImage');
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const resizedImage = document.createElement("img")

  
  resizedImage.onload = () => {
    console.log("onload resizedImage")
    const a = tf.browser.fromPixels(resizedImage, 4)
    // a.print()
    console.log(a)
    console.log(a.shape)
    generateImage(a)
  } 

  img.onload = () => {
    resizedImage.src = resizeImage(img, 720, 720)
  }
  img.src = img_path;
}

run();
