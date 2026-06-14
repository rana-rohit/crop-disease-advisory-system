const result =
    JSON.parse(
        localStorage.getItem(
            "predictionResult"
        )
    );

if (!result) {

    window.location.href =
        "index.html";
}


document.getElementById(
    "diseaseName"
).textContent =
    result.prediction.disease;


document.getElementById(
    "confidence"
).textContent =
    `Confidence: ${result.prediction.confidence}%`;


const list =
    document.getElementById(
        "topPredictions"
    );


result.top_predictions.forEach(
    prediction => {

        const item =
            document.createElement(
                "li"
            );

        item.textContent =
            `${prediction.disease} (${prediction.confidence}%)`;

        list.appendChild(
            item
        );
    }
);