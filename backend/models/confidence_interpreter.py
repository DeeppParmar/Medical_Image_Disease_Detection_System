"""
Central Confidence Interpretation Layer
========================================

Single reusable function for interpreting model confidence across ALL
models (MURA, CheXNet, TB, RSNA). Provides consistent, clinically-safe
labels instead of misleading "Detected" for low-confidence results.

Usage:
    from models.confidence_interpreter import interpret_confidence, enrich_result
    
    # Enrich a single result dict in-place
    enrich_result(result_dict)
"""


def interpret_confidence(confidence_pct: int) -> dict:
    """
    Interpret a confidence percentage into clinically meaningful labels.

    Args:
        confidence_pct: Integer 0-100 representing model confidence.

    Returns:
        dict with keys:
            confidence_level:  "High" | "Moderate" | "Low" | "Minimal"
            risk_status:       "high_risk" | "moderate" | "possible" | "unlikely"
            status_label:      Human-readable status string
            final_assessment:  Clinical assessment message
            display_status:    Frontend status mapping ("critical" | "warning" | "healthy")
    """
    if confidence_pct >= 75:
        return {
            'confidence_level': 'High',
            'risk_status': 'high_risk',
            'status_label': 'High risk abnormality detected',
            'final_assessment': 'High confidence detection — medical attention recommended',
            'display_status': 'critical',
        }
    elif confidence_pct >= 50:
        return {
            'confidence_level': 'Moderate',
            'risk_status': 'moderate',
            'status_label': 'Moderate probability detected',
            'final_assessment': 'Moderate confidence — further evaluation advised',
            'display_status': 'warning',
        }
    elif confidence_pct >= 30:
        return {
            'confidence_level': 'Low',
            'risk_status': 'possible',
            'status_label': 'Possible abnormality (low confidence)',
            'final_assessment': 'Inconclusive — further evaluation recommended',
            'display_status': 'warning',
        }
    else:
        return {
            'confidence_level': 'Minimal',
            'risk_status': 'unlikely',
            'status_label': 'Unlikely abnormality',
            'final_assessment': 'No strong evidence detected',
            'display_status': 'healthy',
        }


def enrich_result(result: dict) -> dict:
    """
    Enrich a prediction result dict with confidence interpretation fields.

    Adds: confidence_level, final_assessment, and updates status based
    on the confidence-based interpretation (only for non-healthy results).

    Args:
        result: Dict with at least 'confidence' and 'status' keys.

    Returns:
        The same dict, mutated with new fields added.
    """
    confidence = result.get('confidence', 0)
    current_status = result.get('status', 'healthy')

    interpretation = interpret_confidence(confidence)

    # Add confidence level (applies to all results)
    result['confidence_level'] = interpretation['confidence_level']

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
        # ABNORMAL results get risk-based assessments
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
