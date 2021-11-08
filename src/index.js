import React from "react";
import ReactDOM from "react-dom";
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';
import "./index.css";
tf.setBackend('webgl');

const threshold = 0.75;

async function load_model() {
    const model = await loadGraphModel("https://raw.githubusercontent.com/AzlinaZanariahMalik/FSL-Phrase-Detection/master/models/fsl-phrase-detector/model.json");
    return model;
  }

let classesDir = {
   
    1: {
      name:'A',
      id:1,
    },
    2: {
      
      name:'Aking',
      id:2
    },
    3: {
      name:'Ako',
      id:3,
    },
    4: {
      name:'Ano',
      id:4,
    },
    5: {
      name:'Ayos Lang',
      id:5,
    },
    6: {
      name:'B',
      id:6,
    },
    7: {   
      name:'Baka',
      id:7,
    },
    8: {
      id:8,
      name:'C'
    },
    9: {
      name:'D',
      id:9,
    },
    10: {
      name:'E',
      id:10,
    },
    11: {
      name:'F',
      id:11,
    },
    12: {
      name:'G',
      id:12,
    },
    13: {
      name:'Gabi',
      id:13,
    },
    14: {
      name:'H',
      id:14,
    },
    15: {
      name:'Hapon',
      id:15,
    },
    16: {
      name:'Hi',
      id:16,
    },
    17: {
      name:'Hindi',
      id:17
    },
    18: {
      name:'Hintay',
      id:18
    },
    19: {
      name:'I',
      id:19,
    },
    20: {
      name:'Ikinagagalak',
      id:20
    },
    21: {
      name:'J',
      id:21
    },
    22: {
      name:'K',
      id:22,
    },
    23: {
      name:'Ka',
      id:23,
    },
    24: {
      name:'Kamusta',
      id:24,
    },
    25: {
      name:'L',
      id:25,
    },
    26: {
      name:'M',
      id:26,
    },
    27: {
      name:'Mabagal',
      id:27,
    },
    28: {
      name:'Magandang',
      id:28,
    },
    29: {
      name:'Magkita',
      id:29,
    },
    30: {
      name:'Mahal Kita',
      id:30,
    },
    31: {
      name:'Makilala',
      id:31,
    },
    32: {
      name:'Mamaya',
      id:32
    },
    33: {
      name:'N',
      id:33,
    },
    34: {
      name:'Ngayon',
      id:34,
    },
    35: {
      name:'Nye',
      id:35,
    },
    36: {
      name:'O',
      id:36,
    },
    37: {
      name:'Oo',
      id:37,
    },
    38: {
      name:'Oras na',
      id:38,
    },
    39: {
      name:'P',
      id:39,
    },
    40: {
      name:'Paalam',
      id:40,
    },
    41: {
      name:'Pakiulit',
      id:41,
    },
    42: {
      name:'Pakiusap',
      id:42,
    },
    43: {
      name:'Pangalan',
      id:43,
    },
    44: {
      name:'Paumanhin',
      id:44,
    },
    45: {
      name:'Q',
      id:45,
    },
    46: {
      name:'R',
      id:46,
    },
    47: {
      name:'S',
      id:47,
    },
    48: {
      name:'Salamat',
      id:48,
    },
    49: {
      name:'T',
      id:49,
    },
    50: {
      name:'Tanghali',
      id:50,
    },
    51: {
      name:'U',
      id:51,
    },
    52: {
      name:'Umaga',
      id:52,
    },
    53: {
      name:'V',
      id:53,
    },
    54: {
      name:'W',
      id:54,
    },
    55: {
      name:'Walang anuman',
      id:55,
    },
    56: {
      name:'X',
      id:56,
    },
    57: {
      name:'Y',
      id:57,
    },
    58: {
      name:'Z',
      id:58,
    }
}

class App extends React.Component {
  videoRef = React.createRef();
  canvasRef = React.createRef();


  componentDidMount() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const webCamPromise = navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            facingMode: "user"
          }
        })
        .then(stream => {
          window.stream = stream;
          this.videoRef.current.srcObject = stream;
          return new Promise((resolve, reject) => {
            this.videoRef.current.onloadedmetadata = () => {
              resolve();
            };
          });
        });

      const modelPromise = load_model();

      Promise.all([modelPromise, webCamPromise])
        .then(values => {
          this.detectFrame(this.videoRef.current, values[0]);
        })
        .catch(error => {
          console.error(error);
        });
    }
  }

    detectFrame = (video, model) => {
        tf.engine().startScope();
        model.executeAsync(this.process_input(video)).then(predictions => {
        this.renderPredictions(predictions, video);
        requestAnimationFrame(() => {
          this.detectFrame(video, model);
        });
        tf.engine().endScope();
      });
  };

  process_input(video_frame){
    const tfimg = tf.browser.fromPixels(video_frame).toInt();
    const expandedimg = tfimg.transpose([0,1,2]).expandDims();
    return expandedimg;
  };

  buildDetectedObjects(scores, threshold, boxes, classes, classesDir) {
    const detectionObjects = []
    var video_frame = document.getElementById('frame');

    scores[0].forEach((score, i) => {
      if (score > threshold) {
        const bbox = [];
        const minY = boxes[0][i][0] * video_frame.offsetHeight;
        const minX = boxes[0][i][1] * video_frame.offsetWidth;
        const maxY = boxes[0][i][2] * video_frame.offsetHeight;
        const maxX = boxes[0][i][3] * video_frame.offsetWidth;
        bbox[0] = minX;
        bbox[1] = minY;
        bbox[2] = maxX - minX;
        bbox[3] = maxY - minY;
        detectionObjects.push({
          class: classes[i],
          label: classesDir[classes[i]].name,
          score: score.toFixed(4),
          bbox: bbox
        })
      }
    })
    return detectionObjects
  }

  renderPredictions = predictions => {
    const ctx = this.canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    //Getting predictions
    const boxes = predictions[4].arraySync();
    const scores = predictions[5].arraySync();
    const classes = predictions[6].dataSync();
    const detections = this.buildDetectedObjects(scores, threshold,
                                    boxes, classes, classesDir);

    detections.forEach(item => {
      const x = item['bbox'][0];
      const y = item['bbox'][1];
      const width = item['bbox'][2];
      const height = item['bbox'][3];

      // Draw the bounding box.
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);

      // Draw the label background.
      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(item["label"] + " " + (100 * item["score"]).toFixed(2) + "%").width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    detections.forEach(item => {
      const x = item['bbox'][0];
      const y = item['bbox'][1];

      // Draw the text last to ensure it's on top.
      ctx.fillStyle = "#000000";
      ctx.fillText(item["label"] + " " + (100*item["score"]).toFixed(2) + "%", x, y);
    });
  };

  render() {
    return (
      <div>
        <h1>FSL Phrase Detection</h1>
        <h3>Capstone Reseach</h3>
        <video
          style={{height: '600px', width: "500px"}}
          className="size"
          autoPlay
          playsInline
          muted
          ref={this.videoRef}
          width="600"
          height="500"
          id="frame"
        />
        <canvas
          className="size"
          ref={this.canvasRef}
          width="600"
          height="500"
        />
      </div>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
