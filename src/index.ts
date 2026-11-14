import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

/**
 * Initialization data for the @jupyter-ai/router extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: '@jupyter-ai/router:plugin',
  description: 'Core routing layer of Jupyter AI',
  autoStart: true,
  activate: (app: JupyterFrontEnd) => {
    console.log('JupyterLab extension @jupyter-ai/router is activated!');
  }
};

export default plugin;
