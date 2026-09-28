import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import glob
import argparse


def find_mri_files(oasis_root: str, disc_start: int, disc_end: int) -> list[str]:
    """
    Busca archivos *_t88_gfc.img dentro del rango de discos especificado.
    
    Patrón esperado:
    OASIS/discN/OAS1_XXXX_MR1/PROCESSED/MPRAGE/T88_111/*_t88_gfc.img
    """
    mri_files = []
    for disc_num in range(disc_start, disc_end + 1):
        disc_dir = os.path.join(oasis_root, f"disc{disc_num}")
        if not os.path.isdir(disc_dir):
            print(f"[WARN] No se encontró: {disc_dir}, saltando...")
            continue

        pattern = os.path.join(
            disc_dir, "OAS1_*", "PROCESSED", "MPRAGE", "T88_111", "*_t88_gfc.img"
        )
        found = sorted(glob.glob(pattern))
        print(f"[INFO] disc{disc_num}: {len(found)} archivos encontrados")
        mri_files.extend(found)

    return mri_files


def extract_subject_id(filepath: str) -> str:
    """
    Extrae el ID del sujeto desde la ruta del archivo.
    Ej: .../OAS1_0001_MR1/... -> OAS1_0001_MR1
    """
    parts = filepath.replace("\\", "/").split("/")
    for part in parts:
        if part.startswith("OAS1_"):
            return part
    return os.path.basename(filepath)


def process_mri(filepath: str, output_dir: str) -> dict:
    """
    Procesa un archivo MRI:
    - Carga la imagen con nibabel
    - Extrae métricas (dimensiones, dtype, min, max)
    - Guarda el corte axial central como PNG
    
    Retorna un diccionario con las métricas.
    """
    subject_id = extract_subject_id(filepath)

    # Cargar imagen
    img = nib.load(filepath)
    data = img.get_fdata()
    data = np.squeeze(data)  # Eliminar dimensiones extra

    # Métricas
    metrics = {
        "subject_id": subject_id,
        "filename": os.path.basename(filepath),
        "dimensions": str(data.shape),
        "dtype": str(data.dtype),
        "min_value": float(np.min(data)),
        "max_value": float(np.max(data)),
    }

    # Corte axial central y guardado como PNG
    if data.ndim == 3:
        slice_index = data.shape[2] // 2
        slice_data = data[:, :, slice_index].T
    else:
        print(f"[WARN] {subject_id}: shape inesperado {data.shape}, saltando imagen")
        return metrics

    png_filename = f"{subject_id}_axial.png"
    png_path = os.path.join(output_dir, png_filename)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(slice_data, cmap="gray", origin="lower")
    ax.set_title(f"Corte Axial - {subject_id}")
    ax.axis("off")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    metrics["png_path"] = png_path
    return metrics


def save_metrics_csv(metrics_list: list[dict], output_dir: str) -> str:
    """Guarda la lista de métricas en un archivo CSV."""
    csv_path = os.path.join(output_dir, "mri_metrics.csv")
    fieldnames = [
        "subject_id", "filename", "dimensions",
        "dtype", "min_value", "max_value", "png_path",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_list)

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Procesa MRIs del dataset OASIS en batch"
    )
    parser.add_argument(
        "oasis_root",
        help="Ruta raíz de la carpeta OASIS (que contiene disc1, disc2, ...)",
    )
    parser.add_argument(
        "--start", type=int, default=1,
        help="Número del primer disco (default: 1)",
    )
    parser.add_argument(
        "--end", type=int, default=1,
        help="Número del último disco (default: 1)",
    )

    args = parser.parse_args()

    oasis_root = os.path.expanduser(args.oasis_root)
    output_dir = os.path.join(oasis_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"OASIS root: {oasis_root}")
    print(f"Procesando discos: {args.start} a {args.end}")
    print(f"Output: {output_dir}\n")

    # Buscar archivos
    mri_files = find_mri_files(oasis_root, args.start, args.end)

    if not mri_files:
        print("[ERROR] No se encontraron archivos MRI. Verifica la ruta y la estructura de carpetas.")
        return

    print(f"\nTotal de archivos a procesar: {len(mri_files)}\n")

    # Procesar cada archivo
    all_metrics = []
    for i, filepath in enumerate(mri_files, 1):
        subject_id = extract_subject_id(filepath)
        print(f"[{i}/{len(mri_files)}] Procesando {subject_id}...")

        try:
            metrics = process_mri(filepath, output_dir)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"[ERROR] Falló {subject_id}: {e}")

    # Guardar CSV
    csv_path = save_metrics_csv(all_metrics, output_dir)

    print(f"\n{'='*50}")
    print(f"Procesamiento completado")
    print(f"  Imágenes procesadas: {len(all_metrics)}/{len(mri_files)}")
    print(f"  PNGs guardados en:   {output_dir}/")
    print(f"  Métricas en:         {csv_path}")


if __name__ == "__main__":
    main()