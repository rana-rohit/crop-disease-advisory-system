const imageInput =
    document.getElementById(
        "imageInput"
    );

const previewImage =
    document.getElementById(
        "previewImage"
    );

const analyzeButton =
    document.getElementById(
        "analyzeButton"
    );


imageInput.addEventListener(
    "change",
    function () {

        const file =
            imageInput.files[0];

        if (!file) {
            return;
        }

        previewImage.src =
            URL.createObjectURL(file);

        previewImage.style.display =
            "block";
    }
);


analyzeButton.addEventListener(
    "click",
    async function () {

        const file =
            imageInput.files[0];

        if (!file) {

            alert(
                "Please select an image."
            );

            return;
        }

        try {

            const result =
                await predictDisease(
                    file
                );

            localStorage.setItem(
                "predictionResult",
                JSON.stringify(result)
            );

            window.location.href =
                "result.html";

        } catch (error) {

            console.error(error);

            alert(
                "Prediction failed."
            );
        }
    }
);