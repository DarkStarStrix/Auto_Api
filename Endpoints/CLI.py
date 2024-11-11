import click
import requests
import json
import time
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console ()


class AutoMLClient:
    def __init__(self, base_url, api_key=None, token=None):
        self.base_url = base_url.rstrip ('/')
        self.api_key = api_key
        self.token = token

    def authenticate(self):
        response = requests.post (
            f"{self.base_url}/api/v1/token",
            json={"api_key": self.api_key}
        )
        self.token = response.json () ["token"]
        return self.token

    def get_headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    def get_templates(self):
        response = requests.get (
            f"{self.base_url}/api/v1/config/templates",
            headers=self.get_headers ()
        )
        return response.json () ["templates"]

    def start_training(self, config):
        response = requests.post (
            f"{self.base_url}/api/v1/train",
            headers=self.get_headers (),
            json=config
        )
        return response.json ()

    def get_job_status(self, job_id):
        response = requests.get (
            f"{self.base_url}/api/v1/jobs/{job_id}",
            headers=self.get_headers ()
        )
        return response.json ()


@click.group ()
def cli():
    """AutoML CLI - Interact with AutoML API"""
    pass


@cli.command ()
@click.option ('--api-key', prompt=True, hide_input=True)
@click.option ('--save', is_flag=True)
def login(api_key, save):
    """Authenticate with the AutoML API"""
    client = AutoMLClient ("http://localhost:8000", api_key=api_key)
    token = client.authenticate ()
    if save:
        with open ('.automl_config', 'w') as f:
            json.dump ({"token": token}, f)
    console.print ("[green]Successfully authenticated!")


@cli.command ()
def templates():
    """List available model templates"""
    try:
        with open ('.automl_config') as f:
            config = json.load (f)
    except FileNotFoundError:
        console.print ("[red]Please login first using 'automl login --save'")
        return

    client = AutoMLClient ("http://localhost:8000", token=config ["token"])
    templates = client.get_templates ()

    table = Table (title="Available Templates")
    table.add_column ("Template Name")
    table.add_column ("Parameters")

    for name, details in templates.items ():
        table.add_row (name, json.dumps (details ["parameters"], indent=2))

    console.print (table)


@cli.command ()
@click.option ('--template', prompt=True)
@click.option ('--data-path', prompt=True)
@click.option ('--target', prompt=True)
def train(template, data_path, target):
    """Start a new training job"""
    try:
        with open ('.automl_config') as f:
            config = json.load (f)
    except FileNotFoundError:
        console.print ("[red]Please login first using 'automl login --save'")
        return

    client = AutoMLClient ("http://localhost:8000", token=config ["token"])
    templates = client.get_templates ()

    if template not in templates:
        console.print (f"[red]Template '{template}' not found!")
        return

    training_config = {
        "model_type": template,
        "config": templates [template] ["parameters"],
        "data_config": {
            "input_path": data_path,
            "target_column": target
        }
    }

    response = client.start_training (training_config)
    job_id = response ["job_id"]

    with Progress () as progress:
        task = progress.add_task ("[cyan]Training...", total=None)
        while True:
            status = client.get_job_status (job_id)
            if status ["status"] in ["completed", "failed"]:
                break
            time.sleep (5)

    if status ["status"] == "completed":
        console.print ("[green]Training completed successfully!")
        console.print ("Metrics:", status ["metrics"])
    else:
        console.print ("[red]Training failed!")
        console.print ("Error:", status.get ("error", "Unknown error"))


if __name__ == '__main__':
    cli ()
