#!/usr/bin/env python3
"""
Script principal pour l'analyse d'épis de blé avec YOLO OBB

Utilisation:
    python src/main_obb.py <image_ou_dossier> [options]

Exemples:
    python src/main_obb.py data/raw/GOPR2583.JPG
    python src/main_obb.py data/raw/ --batch
    python src/main_obb.py data/test_20/ --output output/test_obb
"""

import argparse
import logging
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent))

from analyzer_obb import create_analyzer_from_config, WheatSpikeAnalyzerOBB

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wheat_analyzer_obb.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('WheatAnalyzerOBB')


def main():
    parser = argparse.ArgumentParser(
        description='Analyse d\'épis de blé avec YOLO OBB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline d'analyse:
  1. Détection YOLO OBB : règle, épis (spike/whole_spike), sachets
  2. Calibration : calcul px/mm depuis la règle
  3. Mesures morphométriques : longueur, largeur, aire, périmètre
  4. Comptage des épillets : modèle YOLO dédié
  5. Identification du sachet : OCR des chiffres (bac-ligne-colonne)
  6. Export : JSON + images de debug

Images de debug générées:
  - 01_detections_obb.png    : Toutes les détections OBB
  - 02_ruler_calibration.png : Calibration avec la règle
  - 03_spikes_analysis.png   : Épis avec mesures et épillets
  - 04_bag_identification.png: Sachet avec OCR
  - 05_final_result.png      : Résultat final annoté
        """
    )
    
    parser.add_argument(
        'input',
        help='Chemin vers l\'image ou dossier d\'images'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Chemin vers le fichier de configuration (default: config/config.yaml)'
    )
    parser.add_argument(
        '--output',
        default='output',
        help='Dossier de sortie (default: output)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Traiter plusieurs images (mode batch)'
    )
    parser.add_argument(
        '--no-debug',
        action='store_true',
        help='Désactiver les images de debug'
    )
    parser.add_argument(
        '--low-debug',
        action='store_true',
        help='Mode debug léger: génère uniquement result_annotated.png'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Reprendre le batch en sautant les images déjà traitées'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Afficher plus de détails'
    )
    
    args = parser.parse_args()
    
    # Niveau de log
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("🌾 Wheat Spike Analyzer - YOLO OBB")
    logger.info("=" * 60)
    
    # Vérifier le fichier de config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration non trouvée: {config_path}")
        logger.info("Créez le fichier ou utilisez --config pour spécifier un autre fichier")
        return 1
    
    # Créer l'analyseur
    try:
        # Déterminer le niveau de debug
        if args.no_debug:
            debug = False
        elif args.low_debug:
            debug = 'low'
        else:
            debug = True
        
        analyzer = create_analyzer_from_config(
            config_path=str(config_path),
            output_dir=args.output,
            debug=debug
        )
    except Exception as e:
        logger.error(f"Erreur d'initialisation: {e}")
        return 1
    
    # Traiter les images
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        # Mode batch
        if not input_path.is_dir():
            logger.error(f"Le chemin n'est pas un dossier: {input_path}")
            return 1
        
        # Collecter les images
        image_paths = []
        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']:
            image_paths.extend(input_path.glob(ext))
        
        image_paths = sorted(set(image_paths))
        
        if not image_paths:
            logger.error(f"Aucune image trouvée dans: {input_path}")
            return 1
        
        # Filtrer les images déjà traitées si --resume
        if args.resume:
            output_path = Path(args.output)
            already_done = set()
            for results_file in output_path.glob('**/results.json'):
                folder_name = results_file.parent.name
                already_done.add(folder_name)
            
            original_count = len(image_paths)
            image_paths = [p for p in image_paths if p.stem not in already_done]
            skipped = original_count - len(image_paths)
            
            if skipped > 0:
                logger.info(f"Mode reprise: {skipped} images déjà traitées, {len(image_paths)} restantes")
            
            if not image_paths:
                logger.info("Toutes les images ont déjà été traitées!")
                # Générer quand même le CSV final
                results = []
                for results_file in output_path.glob('**/results.json'):
                    import json
                    with open(results_file, 'r') as f:
                        results.append(json.load(f))
                if results:
                    analyzer.export_batch_csv(results)
                return 0
        
        logger.info(f"Traitement de {len(image_paths)} images...")
        
        results = analyzer.analyze_batch([str(p) for p in image_paths])
        
        logger.info(f"\n{'='*60}")
        logger.info("RÉSUMÉ")
        logger.info(f"{'='*60}")
        logger.info(f"Images traitées: {len(results)}/{len(image_paths)}")
        
        # Statistiques
        total_spikes = sum(r.get('spike_count', 0) for r in results)
        calibrated = sum(1 for r in results if r.get('calibration', {}).get('ruler_detected'))
        identified = sum(1 for r in results if r.get('bag', {}).get('sample_id'))
        
        logger.info(f"Total épis détectés: {total_spikes}")
        logger.info(f"Images calibrées: {calibrated}/{len(results)}")
        logger.info(f"Sachets identifiés: {identified}/{len(results)}")
        
    else:
        # Mode image unique
        if not input_path.exists():
            logger.error(f"Image non trouvée: {input_path}")
            return 1
        
        result = analyzer.analyze_image(str(input_path))
        
        if result:
            logger.info(f"\n{'='*60}")
            logger.info("RÉSULTAT")
            logger.info(f"{'='*60}")
            
            # Afficher le résumé
            cal = result.get('calibration', {})
            if cal.get('ruler_detected'):
                logger.info(f"✓ Calibration: {cal['pixel_per_mm']:.3f} px/mm")
            else:
                logger.info("✗ Pas de calibration (règle non détectée)")
            
            logger.info(f"✓ Épis détectés: {result.get('spike_count', 0)}")
            
            for spike in result.get('spikes', []):
                m = spike.get('measurements', {})
                length = m.get('length_mm') or m.get('length_pixels', 0)
                unit = 'mm' if 'length_mm' in m else 'px'
                spikelet = spike.get('spikelet_count') or '?'
                logger.info(f"  Épi #{spike['id']}: L={length:.1f}{unit}, Épillets={spikelet}")
            
            bag = result.get('bag', {})
            if bag.get('sample_id'):
                logger.info(f"✓ Échantillon: {bag['sample_id']}")
            elif bag.get('detected'):
                logger.info("✗ Sachet détecté mais chiffres non lus")
            else:
                logger.info("✗ Aucun sachet détecté")
            
            output_dir = Path(args.output) / input_path.stem
            logger.info(f"\n📁 Résultats: {output_dir}")
        else:
            logger.error("Échec de l'analyse")
            return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
