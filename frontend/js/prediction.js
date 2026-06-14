function formatDiseaseName(
    diseaseName
) {

    return diseaseName
        .replaceAll(
            "___",
            " → "
        )
        .replaceAll(
            "_",
            " "
        );
}

const uploadedImage =
    localStorage.getItem(
        "uploadedImage"
    );

if (uploadedImage) {

    document.getElementById(
        "uploadedImage"
    ).src =
        uploadedImage;
}

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

const formattedDisease =
    formatDiseaseName(
        result.prediction.disease
    );

document.getElementById(
    "diseaseName"
).textContent =
    formattedDisease;


document.getElementById(
    "confidence"
).textContent =
    `Confidence: ${result.prediction.confidence}%`;

document.getElementById(
    "symptoms"
).textContent =
    result.advisory.symptoms;


document.getElementById(
    "treatment"
).textContent =
    result.advisory.treatment;


document.getElementById(
    "prevention"
).textContent =
    result.advisory.prevention;

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
            `${formatDiseaseName(
                prediction.disease
            )} (${prediction.confidence}%)`;

        list.appendChild(
            item
        );
    }
);