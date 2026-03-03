"""
Central Confidence Interpretation Layer
========================================

Single reusable function for interpreting model confidence across ALL
models (MURA, CheXNet, TB, RSNA). Provides consistent, clinically-safe
labels instead of misleading direct diagnoses.

Key safety features:
  - Never displays direct diagnosis language
  - Uses "AI Suggests Possible..." framing
  - Adds uncertainty band (±5-10%)
  - Includes mandatory AI disclaimer
  - Maps confidence to likelihood levels (not risk levels)

Usage:
    from models.confidence_interpreter import interpret_confidence, enrich_result
    
    # Enrich a single result dict in-place
    enrich_result(result_dict)
"""


AI_DISCLAIMER = "This is an AI-assisted analysis, not a medical diagnosis. Always consult a qualified healthcare professional."


def interpret_confidence(confidence_pct: int) -> dict:
    """
    Interpret a confidence percentage into clinically safe labels.

    Args:
        confidence_pct: Integer 0-100 representing model confidence.

    Returns:
        dict with keys:
            confidence_level:   "High" | "Moderate" | "Low" | "Minimal"
            likelihood_label:   Human-readable likelihood string
            risk_status:        "high_likelihood" | "moderate_likelihood" | "low_confidence" | "inconclusive"
            status_label:       Safe status string (never "detected")
            final_assessment:   Clinical assessment message with safe language
            display_status:     Frontend status mapping ("critical" | "warning" | "healthy")
            uncertainty_range:  Tuple (low, high) representing ± uncertainty band
            ai_disclaimer:      Mandatory disclaimer string
    """
    # Compute uncertainty band: ±5% for high confidence, ±10% for lower
    if confidence_pct >= 80:
        uncertainty = 5
    elif confidence_pct >= 50:
        uncertainty = 8
    else:
        uncertainty = 10

    uncertainty_low = max(0, confidence_pct - uncertainty)
    uncertainty_high = min(100, confidence_pct + uncertainty)

    if confidence_pct >= 80:
        return {
            'confidence_level': 'High',
            'likelihood_label': 'High Likelihood (AI-based)',
            'risk_status': 'high_likelihood',
            'status_label': 'AI suggests possible abnormality',
            'final_assessment': 'High likelihood finding — professional medical evaluation recommended',
            'display_status': 'critical',
            'uncertainty_range': (uncertainty_low, uncertainty_high),
            'uncertainty_pct': uncertainty,
            'ai_disclaimer': AI_DISCLAIMER,
        }
    elif confidence_pct >= 50:
        return {
            'confidence_level': 'Moderate',
            'likelihood_label': 'Moderate Likelihood (AI-based)',
            'risk_status': 'moderate_likelihood',
            'status_label': 'AI suggests possible finding',
            'final_assessment': 'Moderate likelihood — further clinical evaluation advised',
            'display_status': 'warning',
            'uncertainty_range': (uncertainty_low, uncertainty_high),
            'uncertainty_pct': uncertainty,
            'ai_disclaimer': AI_DISCLAIMER,
        }
    elif confidence_pct >= 30:
        return {
            'confidence_level': 'Low',
            'likelihood_label': 'Low Confidence',
            'risk_status': 'low_confidence',
            'status_label': 'Possible finding (low confidence)',
            'final_assessment': 'Low confidence — inconclusive, further evaluation recommended',
            'display_status': 'warning',
            'uncertainty_range': (uncertainty_low, uncertainty_high),
            'uncertainty_pct': uncertainty,
            'ai_disclaimer': AI_DISCLAIMER,
        }
    else:
        return {
            'confidence_level': 'Minimal',
            'likelihood_label': 'Inconclusive',
            'risk_status': 'inconclusive',
            'status_label': 'No significant finding',
            'final_assessment': 'No strong evidence detected by AI',
            'display_status': 'healthy',
            'uncertainty_range': (uncertainty_low, uncertainty_high),
            'uncertainty_pct': uncertainty,
            'ai_disclaimer': AI_DISCLAIMER,
        }


def enrich_result(result: dict) -> dict:
    """
    Enrich a prediction result dict with confidence interpretation fields.

    Adds: confidence_level, likelihood_label, final_assessment,
          uncertainty_range, ai_disclaimer, and updates status.

    Args:
        result: Dict with at least 'confidence' and 'status' keys.

    Returns:
        The same dict, mutated with new fields added.
    """
    confidence = result.get('confidence', 0)
    current_status = result.get('status', 'healthy')

    interpretation = interpret_confidence(confidence)

    # Add all interpretation fields
    result['confidence_level'] = interpretation['confidence_level']
    result['likelihood_label'] = interpretation['likelihood_label']
    result['uncertainty_range'] = list(interpretation['uncertainty_range'])
    result['uncertainty_pct'] = interpretation['uncertainty_pct']
    result['ai_disclaimer'] = interpretation['ai_disclaimer']

    if current_status == 'healthy':
        # HEALTHY results get positive assessments
        if confidence >= 75:
            result['final_assessment'] = 'No significant abnormalities detected'
        elif confidence >= 50:
            result['final_assessment'] = 'No abnormalities detected — clinical correlation recommended if symptomatic'
        elif confidence >= 30:
            result['final_assessment'] = 'Inconclusive — further evaluation recommended'
        else:
            result['final_assessment'] = 'Low confidence normal — consider re-examination'
    else:
        # ABNORMAL results get safe, likelihood-based assessments
        result['final_assessment'] = interpretation['final_assessment']
        result['status'] = interpretation['display_status']

    return result


def enrich_results(results: list) -> list:
    """
    Enrich a list of prediction results with confidence interpretation.

    Args:
        results: List of result dicts from predict_for_frontend().

    Returns:
        The same list with all dicts enriched.
    """
    for r in results:
        enrich_result(r)
    return results
