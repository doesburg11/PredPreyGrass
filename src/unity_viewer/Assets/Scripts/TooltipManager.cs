using UnityEngine;
using TMPro;

public class TooltipManager : MonoBehaviour
{
    public static TooltipManager Instance { get; private set; }

    public TMP_Text tooltipText;
    public CanvasGroup canvasGroup;
    public RectTransform tooltipTransform; // 🆕 Drag your tooltip panel here

    private void Awake()
    {
        // Singleton pattern
        if (Instance != null && Instance != this)
        {
            Destroy(this.gameObject);
        }
        else
        {
            Instance = this;
        }

        // Hide tooltip on start
        HideTooltip();
    }
    private void Update()
    {
        // Follow the mouse
        if (canvasGroup != null && canvasGroup.alpha > 0f && tooltipTransform != null)
        {
            Vector2 mousePosition = Input.mousePosition;
            tooltipTransform.position = mousePosition + new Vector2(95f, 0f); // offset
        }
    }

    public void ShowTooltip(string message)
    {
        if (tooltipText != null)
            tooltipText.text = message;

        if (canvasGroup != null)
        {
            canvasGroup.alpha = 1f;
            canvasGroup.interactable = true;
            canvasGroup.blocksRaycasts = true;
        }
    }

    public void HideTooltip()
    {
        if (canvasGroup != null)
        {
            canvasGroup.alpha = 0f;
            canvasGroup.interactable = false;
            canvasGroup.blocksRaycasts = false;
        }
    }
}
