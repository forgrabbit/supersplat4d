import { Events } from './events';

type PublicModel = {
    name: string;
    path: string;
    url: string;
};

type PublicModelManifest = {
    models?: PublicModel[];
};

let publicModelsPromise: Promise<PublicModel[]> | null = null;

const fetchPublicModels = async (): Promise<PublicModel[]> => {
    const response = await fetch('./model-manifest.json', {
        cache: 'no-store'
    });

    if (!response.ok) {
        throw new Error(`Failed to load model manifest: ${response.status} ${response.statusText}`);
    }

    const manifest = await response.json() as PublicModelManifest;
    return Array.isArray(manifest.models) ? manifest.models : [];
};

const getPublicModels = () => {
    if (!publicModelsPromise) {
        publicModelsPromise = fetchPublicModels().catch((error) => {
            publicModelsPromise = null;
            throw error;
        });
    }

    return publicModelsPromise;
};

const registerPublicModelEvents = (events: Events) => {
    events.function('models.list', () => {
        return getPublicModels();
    });

    events.function('models.load', async (model: PublicModel) => {
        const resetAccepted = await events.invoke('doc.new');
        if (!resetAccepted) {
            return false;
        }

        const result = await events.invoke('import', [{
            filename: model.path,
            url: model.url
        }]);

        if (result && result.length > 0) {
            events.fire('doc.setName', model.path);
            return true;
        }

        return false;
    });
};

export { registerPublicModelEvents };
export type { PublicModel };
