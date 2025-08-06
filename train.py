import click
import mlflow

@click.command()
@click.option("--alpha", default=0.5, type=float)
def train(alpha):
    mlflow.log_param("alpha", alpha)
    # aquí tu lógica de entrenamiento
    mlflow.log_metric("accuracy", 0.8 + alpha)
    print("Entrenamiento finalizado.")

if __name__ == "__main__":
    train()
