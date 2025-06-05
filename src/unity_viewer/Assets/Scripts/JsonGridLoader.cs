using UnityEngine;
using Newtonsoft.Json;

public class JsonGridLoader : MonoBehaviour
{
    [Header("Input File (TextAsset from any folder)")]
    public TextAsset gridJsonFile;

    [Header("Prefabs")]
    public GameObject borderPrefab;
    public GameObject predatorPrefab;
    public GameObject preyPrefab;

    private double[,,] grid;  // [channel, x, y]

    void Start()
    {
        Debug.Log("JsonGridLoader is active");
        if (gridJsonFile == null)
        {
            Debug.LogError("No JSON file assigned to JsonGridLoader.");
            return;
        }

        LoadGridFromJson();

        int width = grid.GetLength(1);
        int height = grid.GetLength(2);

        SpawnBorder(width, height);
        SpawnPredators();
        SpawnPrey();
    }

    void LoadGridFromJson()
    {
        double[][][] raw = JsonConvert.DeserializeObject<double[][][]>(gridJsonFile.text);

        int channels = raw.Length;
        int width = raw[0].Length;
        int height = raw[0][0].Length;

        grid = new double[channels, width, height];

        for (int c = 0; c < channels; c++)
            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++)
                    grid[c, x, y] = raw[c][x][y];

        Debug.Log($"Loaded grid: {channels} channels, size {width} x {height}");
    }

    void SpawnBorder(int width, int height)
    {
        for (int x = -1; x <= width; x++)
        {
            for (int y = -1; y <= height; y++)
            {
                if (x == -1 || x == width || y == -1 || y == height)
                {
                    Vector3 position = new Vector3(x, y, 0);
                    Instantiate(borderPrefab, position, Quaternion.identity);
                }
            }
        }
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
                if (energy > 0.0)
                {
                    Vector3 position = new Vector3(x, y, 0);
                    Instantiate(predatorPrefab, position, Quaternion.identity);
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
                if (energy > 0.0)
                {
                    Vector3 position = new Vector3(x, y, 0);
                    Instantiate(preyPrefab, position, Quaternion.identity);
                }
            }
        }
    }
}
