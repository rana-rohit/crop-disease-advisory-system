const API_URL = "http://127.0.0.1:5000/predict";

async function predictDisease(imageFile) {

    const formData = new FormData();

    formData.append(
        "image",
        imageFile
    );

    const response = await fetch(
        API_URL,
        {
            method: "POST",
            body: formData,
        }
    );

    if (!response.ok) {

        throw new Error(
            "Prediction failed"
        );
    }

    return await response.json();
}