using UnityEngine;
using UnityEngine.UI;
using Newtonsoft.Json;
using System.IO;
using TMPro;

public class JsonGridStepPlayer : MonoBehaviour
{
    [Header("Prefabs")]
    public GameObject borderPrefab;
    public GameObject predatorPrefab;
    public GameObject preyPrefab;
    public GameObject grassPrefab;
    public GameObject waterPrefab;
    public TMP_Text stepCounterText;
    public Slider speedSlider;

    [Header("Play Button")]
    public Button playPauseButton;
    public TMP_Text playPauseButtonText;


    [Header("Simulation Settings")]
    public string folderName = "unity_viewer_exports";
    public string filePrefix = "grid_step_";
    public string fileExtension = ".json";
    public int maxSteps = 1000; // Adjust based on your data

    private int currentStep = 1;
    private double[,,] grid;  // [channel, x, y]
    private GameObject[,] activeObjects;
    private bool bordersSpawned = false;
    private int borderWidth = -1;
    private int borderHeight = -1;

    private bool isPlaying = false;
    public float stepDelay = 0.5f; // Delay between steps in seconds
    private float stepTimer = 0f;


    void Start()
    {
        Debug.Log("JsonGridStepPlayer is active");
        LoadGridFromFile(currentStep);
        if (playPauseButtonText != null)
        {
            playPauseButtonText.text = "Play";
        }

    }

    public void TogglePlayPause()
    {
        isPlaying = !isPlaying;

        if (playPauseButtonText != null)
        {
            playPauseButtonText.text = isPlaying ? "Stop" : "Play";
        }

        if (isPlaying)
        {
            stepTimer = 0f;
        }
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.RightArrow))
        {
            NextStep();
        }
        if (Input.GetKeyDown(KeyCode.LeftArrow))
        {
            PreviousStep();
        }

        // Auto-play mode
        if (isPlaying)
        {
            stepTimer += Time.deltaTime;
            if (stepTimer >= stepDelay)
            {
                stepTimer = 0f;
                if (currentStep < maxSteps - 1)
                {
                    currentStep++;
                    LoadGridFromFile(currentStep);
                }
                else
                {
                    isPlaying = false;
                }
            }
        }


    }

    void LateUpdate()
    {
        if (speedSlider != null)
        {
            stepDelay = speedSlider.value;
        }
    }


    public void NextStep()
    {
        if (currentStep < maxSteps - 1)
        {
            currentStep++;
            LoadGridFromFile(currentStep);
        }
    }

    public void PreviousStep()
    {
        if (currentStep > 0)
        {
            currentStep--;
            LoadGridFromFile(currentStep);
        }
    }


    void LoadGridFromFile(int step)
    {
        string filename = $"{filePrefix}{step:00000}{fileExtension}";
        string path = Path.Combine(Application.streamingAssetsPath, folderName, filename);

        if (!File.Exists(path))
        {
            Debug.LogWarning("File not found: " + path);
            return;
        }

        currentStep = step;  // ✅ Only update if file exists
        Debug.Log($"Loading step: {currentStep}");

        // Destroy previous agents
        if (activeObjects != null)
        {
            foreach (var obj in activeObjects)
                if (obj != null) Destroy(obj);
        }

        string json = File.ReadAllText(path);
        double[][][] raw = JsonConvert.DeserializeObject<double[][][]>(json);

        int channels = raw.Length;
        int width = raw[0].Length;
        int height = raw[0][0].Length;

        grid = new double[channels, width, height];
        activeObjects = new GameObject[width, height];

        for (int c = 0; c < channels; c++)
            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++)
                    grid[c, x, y] = raw[c][x][y];

        // Destroy previous predators/prey
        foreach (var obj in FindObjectsByType<GameObject>(FindObjectsSortMode.None))
            if (obj.name.Contains("Predator") || obj.name.Contains("Prey"))
                Destroy(obj);

        // (Re)create borders only if needed
        if (!bordersSpawned || width != borderWidth || height != borderHeight)
        {
            foreach (GameObject obj in Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None))
                if (obj != null && obj.name.Contains("BorderTile"))
                    Destroy(obj);

            SpawnBorder(width, height);
            bordersSpawned = true;
            borderWidth = width;
            borderHeight = height;
        }

        SpawnPredators();
        SpawnPrey();
        SpawnGrass();
        SpawnRiver();
        if (stepCounterText != null)
            stepCounterText.text = $"Step: {step}";
    }

    void SpawnBorder(int width, int height)
    {
        for (int x = -1; x <= width; x++)
        {
            for (int y = -1; y <= height; y++)
            {
                bool isBorder = (x == -1 || x == width || y == -1 || y == height);
                if (isBorder)
                {
                    Vector3 position = new Vector3(x, y, 0);
                    Instantiate(borderPrefab, position, Quaternion.identity);
                }
            }
        }
    }

    float ScaleFromEnergy(double energy)
        {
            float minScale = 0.4f;
            float maxScale = 1.2f;
            float normalized = Mathf.Clamp01((float)energy / 10f);  // adjust divisor as needed
            return Mathf.Lerp(minScale, maxScale, normalized);
        }   

    void SpawnPredators()
    {
        int channel = 1;
        int width = grid.GetLength(1);
        int height = grid.GetLength(2);

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                double energy = grid[channel, x, y];
                if (energy > 0.00)
                {
                    Vector3 position = new Vector3(x, y, 0);
                    GameObject predator = Instantiate(predatorPrefab, position, Quaternion.identity);
                    predator.transform.localScale = Vector3.one * ScaleFromEnergy(energy);

                    var hover = predator.GetComponent<AgentHoverDebugger>();
                    if (hover != null)
                    {
                        hover.energy = (float)energy;
                    }

                    activeObjects[x, y] = predator;
                }
            }
        }
    }

    void SpawnPrey()
    {

        int channel = 2;
        int width = grid.GetLength(1);
        int height = grid.GetLength(2);

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                double energy = grid[channel, x, y];
                if (energy > 0.00)
                {
                    Vector3 position = new Vector3(x, y, 1);
                    GameObject prey = Instantiate(preyPrefab, position, Quaternion.identity);
                    prey.transform.localScale = Vector3.one * ScaleFromEnergy(energy);

                    var hover = prey.GetComponent<AgentHoverDebugger>();
                    if (hover != null)
                    {
                        hover.energy = (float)energy;
                    }

                    activeObjects[x, y] = prey;
                }
            }
        }
    }

    void SpawnGrass()
    {
        int channel = 3;
        int width = grid.GetLength(1);
        int height = grid.GetLength(2);

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                double energy = grid[channel, x, y];
                if (energy > 0.00)
                {
                    Vector3 position = new Vector3(x, y, 2); // background
                    GameObject grass = Instantiate(grassPrefab, position, Quaternion.identity);
                    grass.transform.localScale = Vector3.one * ScaleFromEnergy(energy);

                    var hover = grass.GetComponent<AgentHoverDebugger>();
                    if (hover != null)
                    {
                        hover.energy = (float)energy;
                    }

                    activeObjects[x, y] = grass;
                }
            }
        }
    }

    void SpawnRiver()
    {
        int channel = 4;
        int width = grid.GetLength(1);
        int height = grid.GetLength(2);

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                double water = grid[channel, x, y];
                if (water > 0.00)
                {
                    Vector3 position = new Vector3(x, y, 4); // background
                    GameObject river = Instantiate(waterPrefab, position, Quaternion.identity);
                    activeObjects[x, y] = river;
                }
            }
        }
    }
}
