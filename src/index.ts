import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { requestAPI } from './handler';

/**
 * Initialization data for the @jupyter-ai/router extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: '@jupyter-ai/router:plugin',
  description: 'Core routing layer of Jupyter AI',
  autoStart: true,
  activate: (app: JupyterFrontEnd) => {
    console.log('JupyterLab extension @jupyter-ai/router is activated!');

    requestAPI<any>('health')
      .then(data => {
        console.log(data);
      })
      .catch(reason => {
        console.error(
          `The jupyter_ai_router server extension appears to be missing.\n${reason}`
        );
      });
  }
};

export default plugin;
